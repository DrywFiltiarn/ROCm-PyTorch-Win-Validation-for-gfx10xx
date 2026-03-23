import torch
import torch.nn as nn

def run():
    print(">> Testing VAE Operations (GroupNorm + Upsample) Verbose...")
    channels, res = 512, 64
    print(f"Config: GroupNorm(32, {channels}) | Res: {res} | Mode: FP16")
    
    try:
        gn = nn.GroupNorm(32, channels).cuda().half()
        x = torch.normal(mean=0.5, std=2.0, size=(1, channels, res, res), device="cuda", dtype=torch.float16)
        
        print(f"Input Stats - Min: {x.min().item():.2f}, Max: {x.max().item():.2f}")
        
        print("\n--- Applying GroupNorm & Upsample ---")
        out = gn(x)
        up = nn.Upsample(scale_factor=2, mode='nearest')
        up_out = up(out)
        torch.cuda.synchronize()
        
        has_nan = torch.isnan(up_out).any().item()
        has_inf = torch.isinf(up_out).any().item()

        print(f"Output Statistics:")
        print(f" - Shape: {up_out.shape} | Dtype: {up_out.dtype}")
        print(f" - Mean:  {up_out.mean().item():.4f} | Std: {up_out.std().item():.4f}")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
        print(f" - Inf Check: {'FAIL' if has_inf else 'Pass'}")
        
        if has_nan or has_inf or up_out.std() < 0.01:
            return False

        print("✅ VAE Operations verified.")
        return True
    except Exception as e:
        print(f"❌ VAE Ops Failure: {e}")
        return False