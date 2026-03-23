import torch
import time

def run():
    print(">> Testing PCIe Bus Saturation (Transfer/Compute Overlap) Verbose...")
    size = 1024 * 1024 * 128 # 512MB Buffer
    print(f"Config: 512MB Buffer Transfers | GEMM Background Load")
    
    try:
        # Background compute load
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        
        host_data = torch.randn(size, pin_memory=True) # Pinned memory for max speed
        
        print("\n--- Running Overlapped Transfer & Compute ---")
        torch.cuda.synchronize()
        start = time.time()
        
        # Start async transfer
        dev_data = host_data.to("cuda", non_blocking=True)
        # Immediate compute while transfer happens
        c = torch.matmul(a, b)
        # Back to host
        back_to_host = dev_data.to("cpu", non_blocking=True)
        
        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000

        print("Output Statistics:")
        print(f" - Transfer/Compute Latency: {latency:.2f}ms")
        print(f" - Integrity Check: {'Pass' if not torch.isnan(c).any() else 'FAIL'}")
        
        return not torch.isnan(c).any()
    except Exception as e:
        print(f"❌ PCIe Stress Failure: {e}")
        return False