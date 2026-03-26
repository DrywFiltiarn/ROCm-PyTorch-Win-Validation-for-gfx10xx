import torch
from torchvision.transforms import functional as F
import time

def run():
    print(">> Testing Torchvision Image Kernels...")
    print("Config: 4K Raw -> 1024x1024 Bilinear Resize + 45deg Rotate")
    
    try:
        img = torch.rand(3, 2160, 3840, device="cuda")
        print(f"VRAM Allocated for 4K Image: {torch.cuda.memory_allocated()/1e6:.2f}MB")

        print("Executing Resize & Rotate Kernels...")
        torch.cuda.synchronize()
        start = time.time()
        resized = F.resize(img, [1024, 1024], antialias=True)
        out = F.rotate(resized, 45)
        torch.cuda.synchronize()
        
        latency = (time.time() - start) * 1000
        has_nan = torch.isnan(out).any().item()

        print(f"Output Statistics:")
        print(f" - Shape: {out.shape} | Latency: {latency:.2f}ms")
        print(f" - Mean:  {out.mean().item():.4f}")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
        
        if has_nan or out.mean() == 0:
            print("❌ Error: Vision kernels produced empty or corrupted output.")
            return False
            
        print("✅ Vision transforms verified.")
        return True
    except Exception as e:
        print(f"❌ Vision Ops Failure: {e}")
        return False