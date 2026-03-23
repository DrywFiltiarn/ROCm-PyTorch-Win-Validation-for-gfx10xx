import torch
import time

def run():
    print(">> Testing Matrix Multiplication (GEMM) Performance Verbose...")
    size = 8192
    print(f"Config: {size}x{size} Matrix | Mode: FP16")
    
    try:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        
        # Input Telemetry
        print(f"Input Stats - A Mean: {a.mean().item():.4f} | B Mean: {b.mean().item():.4f}")
        print(f"VRAM Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        print("\n--- Executing FP16 GEMM Kernel ---")
        torch.cuda.synchronize()
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        latency = (time.time() - start)
        
        tflops = (2 * size**3) / latency / 1e12
        has_nan = torch.isnan(c).any().item()
        has_inf = torch.isinf(c).any().item()

        print("Output Statistics:")
        print(f" - Shape: {c.shape} | Performance: {tflops:.2f} TFLOPS")
        print(f" - Mean:  {c.mean().item():.4f} | Std: {c.std().item():.4f}")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
        print(f" - Inf Check: {'FAIL' if has_inf else 'Pass'}")
        
        if has_nan or has_inf:
            return False
            
        print("✅ MatMul computation verified.")
        return True
    except Exception as e:
        print(f"❌ GEMM Failure: {e}")
        return False