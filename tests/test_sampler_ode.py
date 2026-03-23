import torch
import time

def run():
    print(">> Testing Sampler ODE Solver (Euler Iteration) Verbose...")
    steps = 8
    config = f"Steps={steps}, Shape=[1, 4, 64, 64], Mode=FP16"
    print(f"Config: {config}")
    
    try:
        latent = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
        # Simulate high-frequency noise prediction
        noise_pred = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
        sigmas = torch.linspace(10, 0.1, steps, device="cuda", dtype=torch.float16)

        print(f"Input Stats - Latent Mean: {latent.mean().item():.4f} | VRAM: {torch.cuda.memory_allocated()/1e6:.2f}MB")
        print("\n--- Running Iterative Denoising Loop ---")
        
        torch.cuda.synchronize()
        start = time.time()
        for i in range(steps):
            # Euler update: x = x + dt * f(x)
            step_size = sigmas[i]
            latent = latent + noise_pred * step_size
            
            if i % 4 == 0:
                print(f" - Step {i} | Latent Std: {latent.std().item():.4f}")

        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000

        has_nan = torch.isnan(latent).any().item()
        print("Output Statistics:")
        print(f" - Final Shape: {latent.shape} | Total Latency: {latency:.2f}ms")
        print(f" - Final Mean:  {latent.mean().item():.4f} | NaN Check: {'FAIL' if has_nan else 'Pass'}")

        return not has_nan
    except Exception as e:
        print(f"❌ Sampler Failure: {e}")
        return False