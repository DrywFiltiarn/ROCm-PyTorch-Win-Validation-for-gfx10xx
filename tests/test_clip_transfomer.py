import torch
import torch.nn as nn

def run():
    print(">> Testing CLIP Transformer (LayerNorm)...")
    dim = 768
    print(f"Config: Dim={dim} (CLIP-L Standard) | Mode: FP16")

    try:
        ln = nn.LayerNorm(dim).cuda().half()
        tokens = torch.randn(1, 77, dim, device="cuda", dtype=torch.float16) + 10.0 # Shifted mean
        
        print(f"Input Stats - Token Mean: {tokens.mean().item():.4f}")

        print("\n--- Running LayerNorm Inference ---")
        torch.cuda.synchronize()
        out = ln(tokens)
        torch.cuda.synchronize()

        has_nan = torch.isnan(out).any().item()
        print("Output Statistics:")
        print(f" - Out Mean: {out.mean().item():.4f} | Out Std: {out.std().item():.4f}")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")

        return not has_nan and out.std() > 0.01
    except Exception as e:
        print(f"❌ LayerNorm Failure: {e}")
        return False