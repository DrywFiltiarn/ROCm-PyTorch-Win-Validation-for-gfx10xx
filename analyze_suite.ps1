Write-Host "=== Initializing ROCm Validation Environment ===" -ForegroundColor Cyan

$env:MIOPEN_FIND_MODE = "3"
$env:PYTORCH_ROCM_ARCH = "gfx1010"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

$VenvPath = ".\venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    Write-Host ">> Activating Virtual Environment..." -ForegroundColor Yellow
    & $VenvPath
} else {
    Write-Host ">> WARNING: venv not found at $VenvPath. Running with system python." -ForegroundColor Red
}

Write-Host ">> Starting Testsuite Analyzer..." -ForegroundColor Green
python .\testsuite_analyzer.py