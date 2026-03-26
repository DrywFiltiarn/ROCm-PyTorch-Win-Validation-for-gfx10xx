import torch
import time

def run():
    print(">> Testing Multi-Stream Concurrency (Race Condition Stress)...")
    num_streams = 4
    size = 4096
    print(f"Config: {num_streams} Concurrent Streams | {size}x{size} GEMM | Mode: FP16")
    
    try:
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        results = []
        
        print(f"\n--- Launching {num_streams} Parallel Kernels ---")
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for s in streams:
                with torch.cuda.stream(s):
                    a = torch.randn(size, size, device="cuda")
                    b = torch.randn(size, size, device="cuda")
                    results.append(torch.matmul(a, b))
        
        for s in streams: s.synchronize()
        latency = (time.time() - start) * 1000

        has_nan = any(torch.isnan(r).any().item() for r in results)
        
        print("Output Statistics:")
        print(f" - Stream Count: {len(results)} | Total Latency: {latency:.2f}ms")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
        
        return not has_nan
    except Exception as e:
        print(f"❌ Concurrency Failure: {e}")
        return False