import torch
import torch.nn.functional as F

def run():
    print(">> Testing ComfyUI Upscale Kernels...")
    print("Config: Mode=Nearest-Exact, Factor=8x (64->512)")
    
    try:
        low = torch.randn(1, 3, 64, 64, device="cuda")
        print(f"Input Latent VRAM: {low.element_size()*low.nelement()/1e3:.1f}KB")
        
        print("Executing Nearest-Exact Interpolation...")
        torch.cuda.synchronize()
        up = F.interpolate(low, size=(512, 512), mode='nearest-exact')
        torch.cuda.synchronize()
        
        has_nan = torch.isnan(up).any().item()
        print(f"Output Statistics:")
        print(f" - Shape: {up.shape}")
        print(f" - Mean:  {up.mean().item():.4f}")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
        
        if has_nan or up.shape != (1, 3, 512, 512):
            print("❌ Error: Upscale kernel corrupted output or shape.")
            return False
            
        print("✅ Upscale interpolation verified.")
        return True
    except Exception as e:
        print(f"❌ Upscale Failure: {e}")
        return False