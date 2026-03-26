param(
    [switch]$Unattended
)

Write-Host "=== Initializing ROCm Validation Environment ===" -ForegroundColor Cyan

$env:MIOPEN_FIND_MODE = "3"
$env:PYTORCH_ROCM_ARCH = "gfx1010"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

$VenvPath = ".\venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    Write-Host ">> Activating Virtual Environment..." -ForegroundColor Yellow
    try {
        & $VenvPath -ErrorAction SilentlyContinue
    } catch {
        Write-Host ">> Virtual Environment activation encountered a minor issue, continuing..." -ForegroundColor Gray
    }
} else {
    Write-Host ">> WARNING: venv not found at $VenvPath. Running with system python." -ForegroundColor Red
}

Write-Host ">> Starting Testsuite Manager..." -ForegroundColor Green
if ($Unattended) {
    python .\testsuite_manager.py --unattended
} else {
    python .\testsuite_manager.py
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nTestsuite exited with an error (Code: $LASTEXITCODE)" -ForegroundColor Red
    if (-not $Unattended) { Pause }
}