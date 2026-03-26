import torch
import time

def run():
    print(">> Testing Advanced Indexing (Gather)...")
    tokens, dim = 2048, 64
    print(f"Config: 16 Heads, {tokens} Tokens, Dim={dim} | Mode: FP32")
    
    try:
        cache = torch.randn(1, 16, tokens, dim, device="cuda")
        indices = torch.randint(0, tokens, (1, 16, 512), device="cuda")
        
        print("Executing Paged-Gather Indexing...")
        torch.cuda.synchronize()
        start = time.time()
        idx_exp = indices.unsqueeze(-1).expand(-1, -1, -1, dim)
        out = torch.gather(cache, 2, idx_exp)
        torch.cuda.synchronize()
        
        latency = (time.time() - start) * 1000
        has_nan = torch.isnan(out).any().item()

        print(f"Output Statistics:")
        print(f" - Gathered Shape: {out.shape} | Latency: {latency:.2f}ms")
        print(f" - VRAM Peak: {torch.cuda.memory_allocated()/1e6:.2f}MB")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
        
        if has_nan or out.shape != (1, 16, 512, 64):
            return False
            
        print("✅ Indexing kernels verified.")
        return True
    except Exception as e:
        print(f"❌ Indexing Failure: {e}")
        return False