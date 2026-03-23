import torch

def run():
    print(">> Testing Masked Tensor Ops (ControlNet/Inpaint Logic) Verbose...")
    # Simulate a high-res latent mask
    shape = [1, 4, 128, 128]
    print(f"Config: Shape={shape} | Mode: FP16 + Boolean Masking")
    
    try:
        latents = torch.randn(*shape, device="cuda", dtype=torch.float16)
        mask = (torch.rand(*shape, device="cuda") > 0.8) # 20% masked
        
        print(f"Input Stats - Active Pixels: {mask.sum().item()} / {mask.numel()}")

        print("\n--- Executing Masked Fill & Blending ---")
        torch.cuda.synchronize()
        # Simulate inpainting: Replace masked areas with noise
        noise = torch.randn_like(latents)
        out = latents.clone()
        out.masked_fill_(mask, 0.0) # Zero out
        out += (noise * mask.float()) # Inject noise into mask
        torch.cuda.synchronize()

        has_nan = torch.isnan(out).any().item()
        
        print("Output Statistics:")
        print(f" - Output Mean: {out.mean().item():.4f}")
        print(f" - Zero-Value Check: { (out[mask == 0].abs().sum() > 0).item() }")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")

        return not has_nan
    except Exception as e:
        print(f"❌ Masking Failure: {e}")
        return False