# CreditLens 本地集成测试一键脚本（与 CI integration 阶段同序列）
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts/run_integration.ps1
# 可选参数：
#   -SkipCompose   不启动/等待 docker compose（栈已在运行时使用）
#   -KeepStack     测试结束后不停止容器（便于反复调试）
#
# 序列（与 .github/workflows/ci.yml / .gitlab-ci.yml 完全一致）：
#   1) 启动测试栈（PG 5433->5432、Qdrant 6334->6333，数据在 tmpfs，重启即清空）
#   2) 管理身份：创建 NOSUPERUSER NOBYPASSRLS 业务角色 creditlens_app
#   3) 管理身份：Alembic 迁移（env.py 自动剥离 +asyncpg 走同步驱动）
#   4) 管理身份：应用 RLS 基线（ENABLE + FORCE + 租户/案件级策略）
#   5) 管理身份：向业务角色授权（表 owner 为 postgres）
#   6) 业务身份：三案件幂等 Seed
#   7) 业务身份：集成测试（0 skip / 0 fail 门禁）
#
# 关键点：Seed 与测试都以业务角色连接，绝不用超级用户绕过 RLS，
# 否则 RLS 隔离测试形同虚设。

[CmdletBinding()]
param(
    [switch]$SkipCompose,
    [switch]$KeepStack
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Compose = "docker-compose.test.yml"
$AdminUrl = "postgresql://postgres:postgres@localhost:5433/creditlens_test"
$AppUrl = "postgresql+asyncpg://creditlens_app:creditlens_app@localhost:5433/creditlens_test"
$QdrantUrl = "http://localhost:6334"

# uv 可执行文件：优先 PATH，其次 pip --user 安装目录（Windows 常见位置）
# 统一带 --no-sync：本次运行只使用当前 .venv 已装依赖，不隐式改动环境
# （CI 在 before_script 已显式执行 uv sync --frozen，语义一致）
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

# ---------- 1) 测试栈 ----------
if (-not $SkipCompose) {
    Invoke-Step "启动测试栈（PG + Qdrant）" {
        docker compose -f $Compose up -d
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

# ---------- 5) 授权业务角色 ----------
Invoke-Step "向业务角色授权" {
    Invoke-Psql -Sql @"
GRANT USAGE ON SCHEMA public TO creditlens_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO creditlens_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO creditlens_app;
"@
}

# ---------- 6) Seed（业务身份） ----------
Invoke-Step "三案件 Seed（业务角色 + local 对象存储）" {
    $env:DATABASE_URL = $AppUrl
    $env:QDRANT_URL = $QdrantUrl
    $env:OBJECT_STORE_BACKEND = "local"
    & $Uv run --no-sync python scripts/seed_synthetic_data.py
    if ($LASTEXITCODE -ne 0) { throw "Seed 失败" }
}

# ---------- 7) 集成测试（业务身份，0 skip 门禁） ----------
$failed = $false
Invoke-Step "集成测试（真实 PG + Qdrant + RLS）" {
    $env:DATABASE_URL = $AppUrl
    $env:QDRANT_URL = $QdrantUrl
    $env:OBJECT_STORE_BACKEND = "local"
    $output = & $Uv run --no-sync python -m pytest tests/e2e/ -m integration -ra -q --timeout=300 2>&1 | Tee-Object -Variable lines
    $output | ForEach-Object { Write-Host $_ }
    $joined = $lines -join "`n"
    # 门禁与 CI 一致：出现 skipped / failed / error 均视为失败，
    # 且必须确实有用例通过（避免"0 collected"被当成成功）
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

if (-not $KeepStack) {
    Invoke-Step "停止测试栈" {
        docker compose -f $Compose down
    }
}

if ($failed) {
    Write-Host ""
    Write-Host "INTEGRATION FAILED" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "INTEGRATION OK（0 skip / 0 fail）" -ForegroundColor Green
