import torch

def run():
    print(">> Testing LoRA Weight Injection (Patching)...")
    weight = torch.randn(320, 320, device="cuda", dtype=torch.float16)
    lora_a = torch.randn(320, 16, device="cuda", dtype=torch.float16)
    lora_b = torch.randn(16, 320, device="cuda", dtype=torch.float16)
    alpha = 0.5

    print(f"Config: Rank=16, Dim=320 | Alpha={alpha}")

    try:
        print("\n--- Applying LoRA Patch ---")
        torch.cuda.synchronize()
        patch = torch.matmul(lora_a, lora_b) * alpha
        updated_weight = weight + patch
        torch.cuda.synchronize()

        has_nan = torch.isnan(updated_weight).any().item()
        diff = (updated_weight - weight).abs().mean().item()

        print("Output Statistics:")
        print(f" - Mean Delta: {diff:.6f}")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")

        return not has_nan and diff > 0
    except Exception as e:
        print(f"❌ Patching Failure: {e}")
        return False