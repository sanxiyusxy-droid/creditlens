# CreditLens 本地集成测试一键脚本（与 CI integration 阶段同序列）
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts/run_integration.ps1
# 可选参数：
#   -SkipCompose   不启动/等待 docker compose（栈已在运行时使用）
#   -KeepStack     测试结束后不停止容器（便于反复调试）
#   -SkipSync      跳过 uv sync --frozen（确认当前 .venv 已与 uv.lock 一致时使用）
#
# 序列（与 .github/workflows/ci.yml / .gitlab-ci.yml 完全一致）：
#   0) uv sync --frozen（可用 -SkipSync 显式跳过）
#   1) 启动测试栈（PG 5433->5432、Qdrant 6334->6333，数据在 tmpfs，重启即清空）
#   2) 管理身份：创建 NOSUPERUSER NOBYPASSRLS 业务角色 creditlens_app
#   3) 管理身份：Alembic 迁移（env.py 自动剥离 +asyncpg 走同步驱动）
#   4) 管理身份：应用 RLS 基线（ENABLE + FORCE + 租户/案件级策略）
#   5) 管理身份：创建固定测试 Principal/Case/Membership
#   6) 管理身份：向业务角色授权（表 owner 为 postgres；授权根表只读）
#   7) 业务身份：三案件幂等 Seed
#   8) 业务身份：集成测试（0 skip / 0 fail 门禁）
#
# 关键点：Seed 与测试都以业务角色连接，绝不用超级用户绕过 RLS，
# 否则 RLS 隔离测试形同虚设。

[CmdletBinding()]
param(
    [switch]$SkipCompose,
    [switch]$KeepStack,
    [switch]$SkipSync
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Compose = "docker-compose.test.yml"
$AdminUrl = "postgresql://postgres:postgres@localhost:5433/creditlens_test"
$AppUrl = "postgresql+asyncpg://creditlens_app:creditlens_app@localhost:5433/creditlens_test"
$QdrantUrl = "http://localhost:6334"

# uv 可执行文件：优先 PATH，其次 pip --user 安装目录（Windows 常见位置）。
# 默认先执行 uv sync --frozen；后续 uv run --no-sync 避免每一步重复解析环境。
function Resolve-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "$env:USERPROFILE\AppData\Roaming\Python\Python312\Scripts\uv.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts\uv.exe",
        "$env:USERPROFILE\.local\bin\uv.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
    }
    throw "找不到 uv，可执行文件不在 PATH，也不在常见安装目录"
}
$Uv = Resolve-Uv
Write-Host "uv: $Uv"

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "==== $Name ====" -ForegroundColor Cyan
    & $Body
}

function Invoke-Psql {
    param([string]$Sql, [string]$File)
    if ($File) {
        Get-Content -Raw -Encoding UTF8 $File |
            docker compose -f $Compose exec -T postgres psql -v ON_ERROR_STOP=1 -U postgres -d creditlens_test -f -
    }
    else {
        docker compose -f $Compose exec -T postgres psql -v ON_ERROR_STOP=1 -U postgres -d creditlens_test -c $Sql
    }
    if ($LASTEXITCODE -ne 0) { throw "psql 失败（exit=$LASTEXITCODE）" }
}

function Set-IntegrationEnvironment {
    # 与 CI 使用同一确定性离线配置；显式覆盖开发机 .env 中可能存在的真实模型 API。
    $env:DATABASE_URL = $AppUrl
    $env:QDRANT_URL = $QdrantUrl
    $env:OBJECT_STORE_BACKEND = "local_fs"
    $env:EMBEDDING_PROVIDER = "hash_fallback"
    $env:EMBEDDING_VERSION = "hash-embed-v1"
    $env:EMBEDDING_DIM = "256"
    $env:RERANK_PROVIDER = "disabled"
    $env:LLM_PROVIDER = "disabled"
}

$failed = $false
$stackStarted = $false

try {
    if (-not $SkipSync) {
        Invoke-Step "同步冻结依赖" {
            & $Uv sync --frozen
            if ($LASTEXITCODE -ne 0) { throw "uv sync --frozen 失败" }
        }
    }

    # ---------- 1) 测试栈 ----------
    if (-not $SkipCompose) {
        # 从此处起即使 compose up 部分失败，finally 也应清理可能已创建的资源。
        $stackStarted = $true
        Invoke-Step "启动测试栈（PG + Qdrant）" {
            # tmpfs 只有重建容器才保证全新世界；避免复用曾由真实模型写入的 v2 数据。
            docker compose -f $Compose up -d --force-recreate
            if ($LASTEXITCODE -ne 0) { throw "docker compose up 失败" }
        }
        Invoke-Step "等待 PostgreSQL 就绪" {
            for ($i = 1; $i -le 40; $i++) {
                docker compose -f $Compose exec -T postgres pg_isready -U postgres *> $null
                if ($LASTEXITCODE -eq 0) { Write-Host "PostgreSQL ready"; return }
                Start-Sleep -Seconds 2
            }
            throw "PostgreSQL 等待超时"
        }
        Invoke-Step "等待 Qdrant 就绪" {
            for ($i = 1; $i -le 40; $i++) {
                try {
                    $resp = Invoke-WebRequest -Uri "$QdrantUrl/healthz" -TimeoutSec 3 -UseBasicParsing
                    if ($resp.StatusCode -eq 200) { Write-Host "Qdrant ready"; return }
                }
                catch { Start-Sleep -Seconds 2 }
            }
            throw "Qdrant 等待超时"
        }
    }

    # ---------- 2) 业务角色（NOSUPERUSER NOBYPASSRLS） ----------
    Invoke-Step "创建业务角色 creditlens_app" {
        Invoke-Psql -File "infra/postgres/ci_role.sql"
    }

    # ---------- 3) Alembic 迁移（管理身份） ----------
    Invoke-Step "Alembic 迁移（超级用户）" {
        $env:DATABASE_URL = $AdminUrl
        & $Uv run --no-sync alembic upgrade head
        if ($LASTEXITCODE -ne 0) { throw "alembic upgrade 失败" }
    }

    # ---------- 4) RLS 基线（管理身份） ----------
    Invoke-Step "应用 RLS 基线策略" {
        Invoke-Psql -File "infra/postgres/rls_policies.sql"
    }

    # ---------- 5) 测试授权根（只能由管理身份创建） ----------
    Invoke-Step "创建测试 Principal/Case/Membership（管理身份）" {
        Invoke-Psql -File "infra/postgres/ci_seed_principals.sql"
    }

    # ---------- 6) 授权业务角色 ----------
    Invoke-Step "向业务角色授权" {
        Invoke-Psql -Sql @"
GRANT USAGE ON SCHEMA public TO creditlens_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO creditlens_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO creditlens_app;
REVOKE UPDATE, DELETE ON run_events, human_decisions, report_versions, evidence, artifacts FROM creditlens_app;
REVOKE UPDATE, DELETE ON claims FROM creditlens_app;
GRANT UPDATE (review_status) ON claims TO creditlens_app;
REVOKE INSERT, UPDATE, DELETE ON tenants, app_users, financial_metric_definitions, search_index_versions, alembic_version FROM creditlens_app;
REVOKE INSERT, UPDATE, DELETE ON case_memberships FROM creditlens_app;
REVOKE INSERT, DELETE ON credit_cases FROM creditlens_app;
"@
    }

    # ---------- 7) Seed（业务身份） ----------
    Invoke-Step "三案件 Seed（业务角色 + local 对象存储）" {
        Set-IntegrationEnvironment
        & $Uv run --no-sync python scripts/seed_synthetic_data.py
        if ($LASTEXITCODE -ne 0) { throw "Seed 失败" }
    }

    # ---------- 8) 集成测试（业务身份，0 skip 门禁） ----------
    Invoke-Step "集成测试（真实 PG + Qdrant + RLS）" {
        Set-IntegrationEnvironment
        $output = & $Uv run --no-sync python -m pytest tests/e2e/ -m integration -ra -q --timeout=300 2>&1 | Tee-Object -Variable lines
        $pytestExitCode = $LASTEXITCODE
        $output | ForEach-Object { Write-Host $_ }
        $joined = @($lines) -join "`n"
        # 原生进程退出码是第一门禁；文本检查补充 0 skip 与非空 collection 语义。
        if ($pytestExitCode -ne 0) {
            Write-Host "pytest 退出码为 $pytestExitCode" -ForegroundColor Red
            $script:failed = $true
        }
        if ($joined -match "skipped") {
            Write-Host "集成阶段出现 skipped：视为失败（与 CI 门禁一致）" -ForegroundColor Red
            $script:failed = $true
        }
        if ($joined -match "\d+ (failed|error)") {
            Write-Host "集成测试存在失败用例" -ForegroundColor Red
            $script:failed = $true
        }
        if ($joined -notmatch "\d+ passed") {
            Write-Host "集成测试没有任何通过用例" -ForegroundColor Red
            $script:failed = $true
        }
    }
}
finally {
    # 仅清理由本脚本启动的栈；无论迁移、Seed 或测试在哪一步异常都执行。
    if ($stackStarted -and -not $KeepStack) {
        Write-Host ""
        Write-Host "==== 停止测试栈 ====" -ForegroundColor Cyan
        docker compose -f $Compose down
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "docker compose down 失败（exit=$LASTEXITCODE）"
            $failed = $true
        }
    }
}

if ($failed) {
    Write-Host ""
    Write-Host "INTEGRATION FAILED" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "INTEGRATION OK（0 skip / 0 fail）" -ForegroundColor Green
