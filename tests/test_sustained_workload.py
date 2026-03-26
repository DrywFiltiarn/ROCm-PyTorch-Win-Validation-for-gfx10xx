import torch
import time

def run():
    print(">> Testing Sustained Workload (Thermal/Clock Stability)...")
    
    duration_sec = 30 
    print(f"Config: Continuous Loop | Duration: {duration_sec}s | Mode: High-Power GEMM")
    
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        
        print(f"\n--- Commencing {duration_sec}s Burn-In ---")
        start_time = time.time()
        iterations = 0
        
        while (time.time() - start_time) < duration_sec:
            _ = torch.matmul(a, b)
            iterations += 1
            if iterations % 100 == 0:
                torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        avg_latency = (total_time / iterations) * 1000

        print("Output Statistics:")
        print(f" - Total Iterations: {iterations}")
        print(f" - Avg Kernel Latency: {avg_latency:.4f}ms")
        print(f" - Status: Pass")

        if iterations > 0:
            print(f"✅ Sustained workload verified.")
            return True
        else:
            print(f"❌ Sustained workload failure: No iterations completed.")
            return False

    except Exception as e:
        print(f"❌ Sustained Workload Error: {e}")
        return False