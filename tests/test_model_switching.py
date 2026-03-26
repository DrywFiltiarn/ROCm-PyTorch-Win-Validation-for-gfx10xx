import torch
import time

def run():
    print(">> Testing Model Context Switching (VRAM Purge)...")
    config = "Unet_Sim=2GB, VAE_Sim=500MB | Mode: FP16"
    print(f"Config: {config}")
    
    try:
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated() / 1e6
        print(f"Initial Baseline VRAM: {initial_mem:.2f} MB")

        print("\n--- Loading Simulated Unet (2GB) ---")
        unet = torch.randn(1024, 1024, 1024, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        mid_mem = torch.cuda.memory_allocated() / 1e6
        print(f" - Unet Loaded | Current VRAM: {mid_mem:.2f} MB")

        print("\n--- Purging Unet & Loading VAE + CLIP ---")
        del unet
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        vae = torch.randn(256, 1024, 1024, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        
        final_mem = torch.cuda.memory_allocated() / 1e6
        has_nan = torch.isnan(vae).any().item()

        print("Output Statistics:")
        print(f" - Final Model Shape: {vae.shape}")
        print(f" - Peak VRAM Reclaimed: {mid_mem - initial_mem:.2f} MB")
        print(f" - Residual VRAM: {final_mem:.2f} MB")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")

        if has_nan:
            print("❌ Error: Memory corruption during context switch.")
            return False

        if final_mem >= mid_mem:
            print("❌ Error: VRAM Failed to de-allocate during switch.")
            return False

        print("✅ Model switching and VRAM reclamation verified.")
        return True

    except Exception as e:
        print(f"❌ Switching Failure: {e}")
        return False