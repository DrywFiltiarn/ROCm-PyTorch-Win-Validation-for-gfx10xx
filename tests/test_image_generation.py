import torch
import torch.nn as nn
import torchvision
import time
import os

def run():
    print(">> Testing Generative Image Pipeline + Disk I/O Verbose...")
    steps = 20
    output_path = os.path.join("logs", "test_output_image.png")
    
    try:
        latent = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
        unet = nn.Sequential(
            nn.Conv2d(4, 320, 3, padding=1),
            nn.GroupNorm(32, 320),
            nn.ReLU(),
            nn.Conv2d(320, 4, 3, padding=1)
        ).cuda().half()

        vae = nn.Sequential(
            nn.ConvTranspose2d(4, 3, kernel_size=8, stride=8),
            nn.Sigmoid() 
        ).cuda().half()

        torch.cuda.synchronize()
        for _ in range(steps):
            latent = latent * 0.9 + unet(latent) * 0.1
            
        print("--- Finalizing VAE Decode (Modern Autocast) ---")
        # Updated syntax to resolve FutureWarning
        with torch.amp.autocast('cuda', enabled=False):
            image = vae.float()(latent.float())
            
        torch.cuda.synchronize()
        torchvision.utils.save_image(image, output_path)
        
        has_nan = torch.isnan(image).any().item()
        file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

        print("Output Statistics:")
        print(f" - Pixel Mean:  {image.mean().item():.4f}")
        print(f" - NaN Check:   {'FAIL' if has_nan else 'Pass'}")
        print(f" - Disk Write:  {'Pass' if file_exists else 'FAIL'}")

        return not has_nan and file_exists

    except Exception as e:
        print(f"❌ Image Gen Failure: {e}")
        return False