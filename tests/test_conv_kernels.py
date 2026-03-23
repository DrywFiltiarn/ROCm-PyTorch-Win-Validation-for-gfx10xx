import torch
import torch.nn as nn
import time

def run():
    print(">> Testing Convolutional Kernels (MIOpen) Verbose...")
    # Standard SDXL ResNet block: 320 channels, 3x3 kernel
    channels, res = 320, 64
    print(f"Config: Conv2d({channels}, {channels}, kernel=3) | Mode: FP16")
    
    try:
        conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1).cuda().half()
        x = torch.normal(mean=0.0, std=1.0, size=(1, channels, res, res), device="cuda", dtype=torch.float16)
        
        print(f"Input Stats - Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
        
        print("Executing Conv2d Forward Pass...")
        torch.cuda.synchronize()
        start = time.time()
        out = conv(x)
        torch.cuda.synchronize()
        
        latency = (time.time() - start) * 1000
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()

        print(f"Output Statistics:")
        print(f" - Shape: {out.shape} | Latency: {latency:.2f}ms")
        print(f" - Mean:  {out.mean().item():.4f} | Std: {out.std().item():.4f}")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
        print(f" - Inf Check: {'FAIL' if has_inf else 'Pass'}")

        if has_nan or has_inf or out.std() < 0.01:
            print("❌ Error: Convolution produced invalid numerical results or value collapse.")
            return False
            
        print("✅ Convolution kernels verified.")
        return True
    except Exception as e:
        print(f"❌ MIOpen Critical Failure: {e}")
        return False