# CreditLens v1.6 local synthetic demo: bootstrap -> preflight -> API -> HTTP acceptance -> UI.
# No reset/drop/down command is used. Existing Compose volumes and application facts are preserved.

[CmdletBinding()]
param(
    [switch]$UseConfiguredModels,
    [switch]$SkipCompose,
    [switch]$SkipSync,
    [switch]$SkipHttpAcceptance,
    [switch]$BootstrapOnly,
    [int]$AcceptanceTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"

function Resolve-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = @(
        "$env:USERPROFILE\AppData\Roaming\Python\Python312\Scripts\uv.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts\uv.exe",
        "$env:USERPROFILE\.local\bin\uv.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "UV_NOT_FOUND"
}

function Get-LocalEnvValue {
    param([string]$Name)
    if (-not (Test-Path -LiteralPath ".env.local")) { return "" }
    $prefix = "$Name="
    foreach ($line in Get-Content -LiteralPath ".env.local" -Encoding UTF8) {
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith($prefix)) {
            return $trimmed.Substring($prefix.Length).Trim().Trim('"').Trim("'")
        }
    }
    return ""
}

function Invoke-Checked {
    param([string]$Code, [scriptblock]$Body)
    & $Body
    if ($LASTEXITCODE -ne 0) { throw "$Code (exit=$LASTEXITCODE)" }
}

function Test-ApprovedLocalDockerEndpoint {
    param([Parameter(Mandatory = $true)][string]$Endpoint)

    $normalized = $Endpoint.Trim().Replace('\', '/').ToLowerInvariant()
    return $normalized -in @(
        "npipe:////./pipe/docker_engine",
        "npipe:////./pipe/dockerdesktoplinuxengine"
    )
}

function Assert-LocalDockerContext {
    # DOCKER_HOST overrides the selected context. Reject it before reading any
    # context metadata so a remote daemon is never contacted by this script.
    $dockerHostOverride = [string]$env:DOCKER_HOST
    if (
        -not [string]::IsNullOrWhiteSpace($dockerHostOverride) -and
        -not (Test-ApprovedLocalDockerEndpoint -Endpoint $dockerHostOverride)
    ) {
        throw "REMOTE_DOCKER_CONTEXT_FORBIDDEN"
    }

    $contextRows = @(docker context show 2>$null)
    if ($LASTEXITCODE -ne 0 -or $contextRows.Count -ne 1) {
        throw "DOCKER_CONTEXT_UNAVAILABLE"
    }
    $contextName = ([string]$contextRows[0]).Trim()
    if ([string]::IsNullOrWhiteSpace($contextName)) {
        throw "DOCKER_CONTEXT_UNAVAILABLE"
    }

    $endpointRows = @(
        docker context inspect --format '{{.Endpoints.docker.Host}}' $contextName 2>$null
    )
    if ($LASTEXITCODE -ne 0 -or $endpointRows.Count -ne 1) {
        throw "DOCKER_CONTEXT_UNAVAILABLE"
    }
    $endpoint = ([string]$endpointRows[0]).Trim()
    if (-not (Test-ApprovedLocalDockerEndpoint -Endpoint $endpoint)) {
        throw "REMOTE_DOCKER_CONTEXT_FORBIDDEN"
    }
    return $contextName
}

function Wait-Postgres {
    foreach ($attempt in 1..45) {
        docker compose exec -T postgres pg_isready -U creditlens -d creditlens *> $null
        if ($LASTEXITCODE -eq 0) { return }
        Start-Sleep -Seconds 2
    }
    throw "POSTGRES_READINESS_TIMEOUT"
}

function Wait-HttpReady {
    param(
        [string]$Uri,
        [string]$Code,
        [System.Diagnostics.Process]$Process = $null
    )
    foreach ($attempt in 1..45) {
        if ($null -ne $Process) {
            $Process.Refresh()
            if ($Process.HasExited) { throw "API_PROCESS_EXITED" }
        }
        try {
            $response = Invoke-WebRequest -Uri $Uri -TimeoutSec 3 -UseBasicParsing
            if ($response.StatusCode -eq 200) { return }
        }
        catch {}
        Start-Sleep -Seconds 2
    }
    throw "$Code"
}

function Wait-MinioInit {
    foreach ($attempt in 1..45) {
        $containerId = docker compose ps -a -q minio-init 2>$null
        if ($LASTEXITCODE -eq 0 -and $containerId) {
            $state = docker inspect --format "{{.State.Status}} {{.State.ExitCode}}" $containerId 2>$null
            if ($state -eq "exited 0") { return }
            if ($state -match "^exited [1-9]") { throw "MINIO_INIT_FAILED" }
        }
        Start-Sleep -Seconds 2
    }
    throw "MINIO_INIT_TIMEOUT"
}

$DockerContextName = Assert-LocalDockerContext
Write-Host "Using approved local Docker context: $DockerContextName" -ForegroundColor DarkGray

$Uv = Resolve-Uv
if (-not (Test-Path -LiteralPath ".env.local")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env.local"
    Write-Host "Created .env.local from the safe local template." -ForegroundColor Yellow
}

# Infrastructure and identity are always pinned to the local Compose demo. This prevents an
# accidentally configured remote database/object store from receiving bootstrap writes.
$appPassword = $env:APP_DB_PASSWORD
if (-not $appPassword) { $appPassword = Get-LocalEnvValue "APP_DB_PASSWORD" }
if (-not $appPassword) {
    # Keep the role password aligned with an existing local runtime DSN.  This
    # supports older .env.local files that predate APP_DB_PASSWORD without ever
    # printing or forwarding the secret to a remote endpoint.
    $localDatabaseUrl = Get-LocalEnvValue "DATABASE_URL"
    if (
        $localDatabaseUrl -match `
            '^postgresql(?:\+asyncpg)?://creditlens_app:([^@]+)@(localhost|127\.0\.0\.1):5432/creditlens(?:\?.*)?$'
    ) {
        $appPassword = [System.Uri]::UnescapeDataString($Matches[1])
    }
}
if (-not $appPassword) { $appPassword = "creditlens_app" }
$encodedPassword = [System.Uri]::EscapeDataString($appPassword)
$runtimeDatabaseUrl = "postgresql+asyncpg://creditlens_app:${encodedPassword}@localhost:5432/creditlens"
$adminDatabaseUrl = "postgresql+asyncpg://creditlens:creditlens@localhost:5432/creditlens"

$env:APP_ENV = "local"
$env:API_IDENTITY_MODE = "demo"
$env:ALLOW_INSECURE_DEMO_IDENTITY = "true"
$env:APP_DB_PASSWORD = $appPassword
$env:DATABASE_URL = $runtimeDatabaseUrl
$env:QDRANT_URL = "http://localhost:6333"
$env:OBJECT_STORE_BACKEND = "minio"
$env:MINIO_ENDPOINT = "localhost:9000"
$env:MINIO_SECURE = "false"
$env:MINIO_ACCESS_KEY = "creditlens"
$env:MINIO_SECRET_KEY = "creditlens-dev-secret"
$env:MINIO_RAW_BUCKET = "creditlens-raw"
$env:MINIO_DERIVED_BUCKET = "creditlens-parsed"
$env:MINIO_RENDERED_BUCKET = "creditlens-rendered"
$env:TELEMETRY_OUTBOX_WORKER_ENABLED = "true"
$env:TELEMETRY_EXPORTER_BACKEND = "local_directory"
$env:TELEMETRY_LOCAL_DIRECTORY = "evaluation/reports/local/telemetry_delivery"
$env:TELEMETRY_EXPORT_POLL_SECONDS = "0.25"

if (-not $UseConfiguredModels) {
    # Deterministic and non-networked by default. Synthetic data leaves the machine only when
    # -UseConfiguredModels is explicitly supplied.
    $env:LLM_PROVIDER = "disabled"
    $env:LLM_API_BASE = ""
    $env:LLM_API_KEY = ""
    $env:LLM_MODEL = ""
    $env:EMBEDDING_PROVIDER = "hash_fallback"
    $env:EMBEDDING_VERSION = "hash-embed-v1"
    $env:EMBEDDING_DIM = "256"
    $env:EMBEDDING_API_BASE = ""
    $env:EMBEDDING_API_KEY = ""
    $env:EMBEDDING_MODEL = ""
    $env:RERANK_PROVIDER = "lexical_fallback"
    $env:RERANK_API_BASE = ""
    $env:RERANK_API_KEY = ""
    $env:RERANK_MODEL = ""
    $env:QA_ALLOW_EXTRACTIVE_FALLBACK = "true"
}

if (-not $SkipSync) {
    Write-Host "[1/9] Syncing frozen dependencies..." -ForegroundColor Cyan
    Invoke-Checked "UV_SYNC_FAILED" { & $Uv sync --frozen }
}
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "PYTHON_RUNTIME_NOT_FOUND" }

if (-not $SkipCompose) {
    Write-Host "[2/9] Starting local PostgreSQL, Qdrant, MinIO and Redis..." -ForegroundColor Cyan
    Invoke-Checked "COMPOSE_UP_FAILED" {
        docker compose up -d postgres qdrant minio minio-init redis
    }
}
else {
    Write-Host "[2/9] Reusing explicitly requested local Compose stack." -ForegroundColor Cyan
}

Write-Host "[3/9] Waiting for infrastructure readiness..." -ForegroundColor Cyan
Wait-Postgres
Wait-HttpReady -Uri "http://127.0.0.1:6333/healthz" -Code "QDRANT_READINESS_TIMEOUT"
Wait-HttpReady -Uri "http://127.0.0.1:9000/minio/health/ready" -Code "MINIO_READINESS_TIMEOUT"
Wait-MinioInit

Write-Host "[4/9] Applying Alembic head, forced RLS and least-privilege grants..." -ForegroundColor Cyan
$env:DATABASE_URL = $adminDatabaseUrl
try {
    Invoke-Checked "ALEMBIC_UPGRADE_FAILED" { & $Uv run --no-sync alembic upgrade head }
    Invoke-Checked "RLS_BOOTSTRAP_FAILED" {
        & $Uv run --no-sync python scripts/apply_rls.py --bootstrap-demo-principals
    }
}
finally {
    $env:DATABASE_URL = $runtimeDatabaseUrl
}

Write-Host "[5/9] Idempotently seeding synthetic cases, indexes and financial facts..." -ForegroundColor Cyan
$bootstrapArgs = @("run", "--no-sync", "python", "scripts/bootstrap_demo.py")
if ($UseConfiguredModels) { $bootstrapArgs += "--allow-configured-models" }
$bootstrapJson = & $Uv @bootstrapArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host $bootstrapJson -ForegroundColor Red
    throw "DEMO_PREFLIGHT_FAILED"
}
Write-Host $bootstrapJson -ForegroundColor Green

if ($BootstrapOnly) {
    Write-Host "Bootstrap and preflight complete; API/UI were not started." -ForegroundColor Green
    exit 0
}

$api = $null
try {
    Write-Host "[6/9] Starting API at http://127.0.0.1:8000 ..." -ForegroundColor Cyan
    $apiPortOccupied = $false
    try {
        $existing = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health/live" `
            -TimeoutSec 2 -UseBasicParsing
        $apiPortOccupied = $existing.StatusCode -eq 200
    }
    catch {}
    if ($apiPortOccupied) {
        throw "API_PORT_ALREADY_IN_USE"
    }
    $api = Start-Process -PassThru -WindowStyle Hidden -FilePath $Python `
        -ArgumentList @(
            "-m", "uvicorn", "apps.api.main:app",
            "--host", "127.0.0.1", "--port", "8000"
        )
    Wait-HttpReady -Uri "http://127.0.0.1:8000/health/ready" `
        -Code "API_READINESS_TIMEOUT" -Process $api

    Write-Host "[7/9] Running two real-workflow synthetic fail-closed cases..." `
        -ForegroundColor Cyan
    $failureJson = & $Uv run --no-sync python scripts/run_fail_closed_cases.py `
        --execute-system `
        --output evaluation/reports/local/v16_fail_closed_system.json
    if ($LASTEXITCODE -ne 0) {
        Write-Host $failureJson -ForegroundColor Red
        throw "FAIL_CLOSED_SYSTEM_ACCEPTANCE_FAILED"
    }
    Write-Host $failureJson -ForegroundColor Green

    if (-not $SkipHttpAcceptance) {
        Write-Host "[8/9] Running black-box HTTP acceptance..." -ForegroundColor Cyan
        $acceptanceProfile = if ($UseConfiguredModels) {
            "configured-models"
        }
        else {
            "deterministic-offline"
        }
        $acceptanceJson = & $Uv run --no-sync python scripts/http_demo_acceptance.py `
            --timeout-seconds $AcceptanceTimeoutSeconds `
            --profile $acceptanceProfile `
            --output evaluation/reports/local/v16_http_acceptance_latest.json
        if ($LASTEXITCODE -ne 0) {
            Write-Host $acceptanceJson -ForegroundColor Red
            throw "HTTP_ACCEPTANCE_FAILED"
        }
        Write-Host $acceptanceJson -ForegroundColor Green
    }
    else {
        Write-Host "[8/9] HTTP acceptance explicitly skipped." -ForegroundColor Yellow
    }

    Write-Host "[9/9] Starting UI at http://127.0.0.1:8501 ..." -ForegroundColor Cyan
    & $Python -m streamlit run apps/demo/streamlit_app.py `
        --server.address 127.0.0.1 --server.port 8501 --server.headless true
    if ($LASTEXITCODE -ne 0) { throw "STREAMLIT_EXITED_WITH_ERROR" }
}
finally {
    if ($null -ne $api -and -not $api.HasExited) {
        Write-Host "Stopping API process $($api.Id)..."
        Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
    }
}
