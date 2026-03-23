Write-Host "=== Initializing ROCm Validation Environment ===" -ForegroundColor Cyan

$env:PYTORCH_ROCM_ARCH = "gfx1010"

# Activate Virtual Environment
$VenvPath = ".\venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    Write-Host ">> Activating Virtual Environment..." -ForegroundColor Yellow
    & $VenvPath
} else {
    Write-Host ">> WARNING: venv not found at $VenvPath. Running with system python." -ForegroundColor Red
}

# Launch the Suite
Write-Host ">> Starting Testsuite Analyzer..." -ForegroundColor Green
python .\testsuite_analyzer.py