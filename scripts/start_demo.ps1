# CreditLens 演示一条命令启动（任务 30）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
# 前置：Docker Compose 服务已启动（docker compose up -d）、.env.local 已配置、
#       演示库已迁移 + seed（评测或 seed 脚本跑过一次即可）。

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"

Write-Host "[1/3] 检查基础设施..." -ForegroundColor Cyan
docker compose ps --format "{{.Name}}: {{.Status}}" 2>$null

Write-Host "[2/3] 启动 API (http://127.0.0.1:8000)..." -ForegroundColor Cyan
$api = Start-Process -PassThru -WindowStyle Hidden py `
    -ArgumentList "-3.12", "-m", "uv", "run", "uvicorn", "apps.api.main:app", "--port", "8000"

# 等待 API 就绪
$ready = $false
foreach ($i in 1..30) {
    Start-Sleep 1
    try {
        $null = Invoke-RestMethod "http://127.0.0.1:8000/health/ready" -TimeoutSec 2
        $ready = $true; break
    } catch {}
}
if (-not $ready) { Write-Host "API 未能就绪，请检查 Docker 与 .env.local" -ForegroundColor Red; Stop-Process -Id $api.Id; exit 1 }
Write-Host "API 就绪 ✓" -ForegroundColor Green

Write-Host "[3/3] 启动演示页 (http://localhost:8501)..." -ForegroundColor Cyan
try {
    py -3.12 -m uv run streamlit run apps/demo/streamlit_app.py --server.port 8501 --server.headless true
} finally {
    Write-Host "停止 API (pid $($api.Id))..."
    Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
}
