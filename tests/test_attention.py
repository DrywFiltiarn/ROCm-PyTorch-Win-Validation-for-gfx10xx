import torch
import torch.nn.functional as F
import time

def run():
    print(">> Testing Scaled Dot-Product Attention (SDPA) Kernels...")
    config = "Batch=1, Heads=16, Len=4096, Dim=64 | Mode: FP16"
    print(f"Config: {config}")
    
    try:
        q = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16)
        
        print(f"Input Stats - Mean: {q.mean().item():.4f} | VRAM: {torch.cuda.memory_allocated()/1e6:.2f}MB")

        print("\n--- Running SDPA (Math Backend) ---")
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            out = F.scaled_dot_product_attention(q, k, v)
            
        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000

        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        
        print("Output Statistics:")
        print(f" - Shape: {out.shape} | Latency: {latency:.2f}ms")
        print(f" - Mean:  {out.mean().item():.4f} | Std: {out.std().item():.4f}")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
        print(f" - Inf Check: {'FAIL' if has_inf else 'Pass'}")

        if has_nan or has_inf or out.std() < 0.001:
            return False

        return True
    except Exception as e:
        print(f"❌ SDPA Failure: {e}")
        return False