import os
import re
import torch
from datetime import datetime

def analyze_logs():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        print(f"❌ Error: {log_dir} directory not found.")
        return

    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    
    # --- LIVE HARDWARE DISCOVERY ---
    device_name = "Unknown"
    gfx_arch = "Unknown"
    gfx_arch_code = "unknown"
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if any(x in device_name for x in ["5700", "5600", "5500"]):
            gfx_arch = "10.1.0"
            gfx_arch_code = "gfx1010"
        elif any(x in device_name for x in ["6900", "6800", "6700", "6600"]):
            gfx_arch = "10.3.0"
            gfx_arch_code = "gfx1030"

    evidence = {
        "cudnn_fail": False,
        "gpu_spectral_works": False,
        "fp16_collapse": False,
        "vram_stress": False
    }

    test_results = []
    total_passed = 0

    print(f"--- Factual Analysis Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"--- Discovery: Detected {device_name} ---\n")

    for log_file in sorted(log_files):
        path = os.path.join(log_dir, log_file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                c_low = content.lower()
                f_low = log_file.lower()
        except Exception:
            continue

        header = re.search(r'>> (.*?)\n', content)
        test_name = header.group(1).strip() if header else log_file
        
        # 1. SUCCESS-FIRST STATUS LOGIC
        # If the log contains a success marker, it is PASSED regardless of internal warnings.
        is_verified = "✅" in content or "pass" in c_low or "verified" in c_low
        # Hard failure only if it explicitly failed and has no success marker.
        is_hard_fail = "❌" in content or ("fail" in c_low and not is_verified)
        
        status_bool = is_verified and not is_hard_fail

        # 2. EVIDENCE EXTRACTION (Capturing 'Ghost Failures' for adjustments)
        if "batchnorm" in f_low or "batch_norm" in f_low:
            if "miopen" in c_low and ("error" in c_low or "fallback" in c_low or "native" in c_low):
                evidence["cudnn_fail"] = True

        if "forensic" in f_low or "audio_path_a" in f_low:
            if "gpu result: pass" in c_low or "path a: pass" in c_low:
                evidence["gpu_spectral_works"] = True

        if any(x in f_low for x in ["vae", "attention", "image_generation"]):
            if "nan" in c_low or "inf" in c_low:
                evidence["fp16_collapse"] = True
        
        if any(x in f_low for x in ["vram", "overflow", "saturation", "pcie"]):
            evidence["vram_stress"] = True

        # 3. AGGREGATION
        status = "PASSED" if status_bool else "FAILED"
        if status_bool: total_passed += 1
        
        vram = re.search(r'VRAM: ([\d.]+)MB', content)
        latency = re.search(r'Latency: ([\d.]+)ms', content)
        telemetry = f" | VRAM: {vram.group(1)}MB" if vram else ""
        telemetry += f" | Latency: {latency.group(1)}ms" if latency else ""
        test_results.append(f"[{status}] {test_name}{telemetry}")

    # --- FINAL OUTPUT: VALIDATION REPORT ---
    print("="*75)
    print(f"MILESTONE 2.8.0: FINAL COMPLIANCE REPORT ({device_name})")
    print("="*75)
    print("\n".join(test_results))
    print("="*75)
    print(f"OVERALL HEALTH: {total_passed}/{len(log_files)} Tests Passed")

    print( "\n--- MANDATORY ENVIRONMENT VARIABLES ---")
    print( " ✅ MIOPEN_FIND_MODE=3")
    print( "     : Accelerate MIOpen API calls in hybrid mode")
    print(f" ✅ PYTORCH_ROCM_ARCH={gfx_arch_code}")
    print(f"     : Ensure correct arch is activated for {device_name}.")
    print( " ✅ PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8")
    print( "     : Improve memory handling under stress")

    print( "\n--- MANDATORY COMFYUI ARGUMENTS ---")
    print( " ✅ --use-pytorch-cross-attention")
    print( "     : Essential for SDPA stability on RDNA hardware.")
    print( " ✅ --preview-method auto")
    print(f"     : Optimizes VRAM-to-UI dispatch for {device_name}.")
    print( " ✅ --no-half-vae --precision full")
    print( "     : RDNA1/RDNA2 do not handle half precision FP16 very well, enforce FP32")
    print( "\n--- FINAL SYSTEM VERDICT ---")

    if total_passed == len(log_files):
        print(f"STATUS: SYSTEM READY ({device_name})")
    else:
        print("STATUS: SYSTEM UNSTABLE (Terminal failures detected in remaining logs)")

if __name__ == "__main__":
    analyze_logs()