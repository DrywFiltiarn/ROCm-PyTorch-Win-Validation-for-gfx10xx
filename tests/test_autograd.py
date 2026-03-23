import torch

def run():
    print(">> Testing Autograd Engine & Backprop Chain Verbose...")
    print("Config: Matrix 4096^2 | Mode: FP32 Gradient Tracking")
    try:
        x = torch.randn(4096, 4096, device="cuda", requires_grad=True)
        print("\n--- Running Forward & Backward Pass ---")
        y = torch.sigmoid(x).pow(2).mean()
        y.backward()
        torch.cuda.synchronize()

        has_nan = torch.isnan(x.grad).any().item()
        print("Gradient Statistics:")
        print(f" - Grad Shape: {x.grad.shape}")
        print(f" - Grad Mean:  {x.grad.mean().item():.6f}")
        print(f" - NaN Check:  {'FAIL' if has_nan else 'Pass'}")
        
        if has_nan or x.grad.abs().sum() == 0:
            print("❌ Error: Gradient chain corrupted or zeroed.")
            return False
        return True
    except Exception as e:
        print(f"❌ Autograd Failure: {e}")
        return False