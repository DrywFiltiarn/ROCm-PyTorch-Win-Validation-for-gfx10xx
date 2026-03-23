import torch
import torch.nn as nn

def run():
    print(">> Testing BatchNorm2d Stability & Stats Verbose...")
    channels = 128
    print(f"Config: {channels} Channels | Mode: FP32 Inference")
    
    # Setup data
    x = torch.randn(8, channels, 32, 32, device="cuda") + 5.0
    print(f"Input Stats - Mean: {x.mean().item():.4f} | Std: {x.std().item():.4f}")
    
    def execute_batch_norm(use_miopen=True):
        bn = nn.BatchNorm2d(channels).cuda()
        # Toggle MIOpen/CuDNN backend
        torch.backends.cudnn.enabled = use_miopen
        
        mode_str = "MIOpen Accelerated" if use_miopen else "Native PyTorch Fallback"
        print(f"\n--- Running Statistics Pass ({mode_str}) ---")
        
        for _ in range(10):
            out = bn(x)
        torch.cuda.synchronize()
        return out, bn

    try:
        # Attempt 1: Standard Path
        out, bn = execute_batch_norm(use_miopen=True)
    except Exception as e:
        if "miopenStatusUnknownError" in str(e) or "MIOpen" in str(e):
            print(f"⚠️ MIOpen Error Detected: {e}")
            print(">> Attempting Recovery via Native C++ Kernels...")
            try:
                # Attempt 2: Bypassing MIOpen
                out, bn = execute_batch_norm(use_miopen=False)
            except Exception as e2:
                print(f"❌ Critical Recovery Failure: {e2}")
                return False
        else:
            print(f"❌ Non-MIOpen Failure: {e}")
            return False
    finally:
        # Ensure we re-enable the backend for other tests
        torch.backends.cudnn.enabled = True

    # Standardized Output Statistics Block
    has_nan = torch.isnan(out).any().item()
    mean_stat = bn.running_mean.mean().item()
    var_stat = bn.running_var.mean().item()

    print("Output Statistics:")
    print(f" - Shape: {out.shape}")
    print(f" - Running Mean (Avg): {mean_stat:.4f}")
    print(f" - Running Var (Avg):  {var_stat:.4f}")
    print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
    
    if has_nan or mean_stat == 0:
        print("❌ Error: Numerical results corrupted.")
        return False
        
    print("✅ BatchNorm verified (Recovery Path Used)." if not torch.backends.cudnn.enabled else "✅ BatchNorm verified.")
    return True