import torch
import time

def run():
    print(">> Testing VRAM Efficiency & Allocator Stability...")
    print("Config: Cycle=500, Shape=Random[256-1024], Mode=FP32")
    
    try:
        initial_mem = torch.cuda.memory_allocated() / 1e6
        print(f"Initial Allocated VRAM: {initial_mem:.2f} MB")
        
        print("\n--- Running High-Frequency Stress Loop ---")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(500):
            size = torch.randint(256, 1024, (1,)).item()
            temp = torch.empty((size, size), device="cuda", dtype=torch.float32).normal_()
            if i % 100 == 0:
                if torch.isnan(temp).any(): return False
            del temp 

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        final_mem = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        
        print("Output Statistics:")
        print(f" - Loop Latency: {(time.time() - start_time)*1000:.2f}ms")
        print(f" - Final Allocated: {final_mem:.2f} MB")
        print(f" - ROCm Reserved:  {reserved:.2f} MB")
        print(f" - NaN Check: Pass")
        
        return final_mem < (initial_mem + 50.0)
    except Exception as e:
        print(f"❌ VRAM Efficiency Failure: {e}")
        return False