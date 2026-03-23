Write-Host "=== Initializing ROCm Validation Environment ===" -ForegroundColor Cyan

# Set Memory Management Flags for Shared Memory Stability
$env:GPU_MAX_ALLOC_PERCENT = "100"
$env:GPU_MAX_HEAP_SIZE = "100"
$env:GPU_SINGLE_ALLOC_PERCENT = "100"

# MIOpen Stability Overrides
$env:MIOPEN_FIND_MODE = "1"
$env:MIOPEN_DEBUG_DISABLE_CDB_LOOKUP = "1"
#$env:MIOPEN_DEBUG_CONVOLUTION_MAX_Z_SIZE = "0"
#$env:MIOPEN_DEBUG_DISABLE_CUBLAS = "1"
#$env:MIOPEN_DEBUG_DISABLE_GCN_ROCM = "1"
#$env:MIOPEN_DEBUG_DISABLE_CUDNN_CONVOLV = "1"

$env:PYTORCH_ROCM_ARCH = "gfx1010"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

# Activate Virtual Environment
$VenvPath = ".\venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    Write-Host ">> Activating Virtual Environment..." -ForegroundColor Yellow
    & $VenvPath
} else {
    Write-Host ">> WARNING: venv not found at $VenvPath. Running with system python." -ForegroundColor Red
}

# Launch the Testsuite
Write-Host ">> Starting Testsuite Manager..." -ForegroundColor Green
python .\testsuite_manager.py

# Keep window open on exit if there's a crash
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nTestsuite exited with an error (Code: $LASTEXITCODE)" -ForegroundColor Red
    Pause
}