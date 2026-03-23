import torch
import os
import time

def run():
    print(">> Testing Model Weight Integrity (I/O & Precision) Verbose...")
    # Simulate a large weight layer (e.g., 4096 x 4096)
    size = 4096
    temp_file = "integrity_check_tmp.pt"
    print(f"Config: Size={size}x{size} | CPU(FP32) -> GPU(FP16) | Path: {temp_file}")
    
    try:
        # 1. Create 'Golden' weights on CPU in Full Precision
        print("\n--- Generating Golden Weights (CPU FP32) ---")
        golden_weights = torch.randn(size, size, dtype=torch.float32)
        golden_mean = golden_weights.mean().item()
        golden_std = golden_weights.std().item()
        
        # 2. Save to disk
        torch.save(golden_weights, temp_file)
        file_size = os.path.getsize(temp_file) / 1e6
        print(f"Saved Checkpoint: {file_size:.2f} MB")

        # 3. Load back to GPU and cast to FP16
        print("--- Loading to VRAM & Casting to FP16 ---")
        torch.cuda.synchronize()
        start = time.time()
        
        loaded_weights = torch.load(temp_file, map_location="cuda", weights_only=True).half()
        
        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000

        # 4. Numerical Comparison
        # We compare the GPU FP16 mean against the CPU FP32 mean
        # Allowance for FP16 precision loss is typically 1e-3
        gpu_mean = loaded_weights.mean().item()
        delta = abs(golden_mean - gpu_mean)
        
        has_nan = torch.isnan(loaded_weights).any().item()
        has_inf = torch.isinf(loaded_weights).any().item()

        print("Output Statistics:")
        print(f" - Load Latency: {latency:.2f}ms")
        print(f" - Golden Mean:  {golden_mean:.6f}")
        print(f" - GPU Mean:     {gpu_mean:.6f}")
        print(f" - Precision Δ:  {delta:.6f}")
        print(f" - NaN Check:    {'FAIL' if has_nan else 'Pass'}")
        print(f" - Inf Check:    {'FAIL' if has_inf else 'Pass'}")

        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Integrity Threshold: Delta should be very small, and no NaNs/Infs
        if has_nan or has_inf or delta > 1e-3:
            print("❌ Error: Weight corruption detected during Load/Cast.")
            return False

        print("✅ Model weight integrity verified.")
        return True

    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"❌ Integrity Test Failure: {e}")
        return False