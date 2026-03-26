import torch
import torchaudio
import time
import os
import soundfile as sf

def run():
    print(">> Forensic Audit: Isolating GPU vs. Mathematical Flaw...")
    n_fft = 1024
    path_gpu = os.path.join("logs", "test_audio_path_A_GPU.wav")
    path_cpu = os.path.join("logs", "test_audio_path_B_CPU.wav")
    
    try:
        spec_gpu = torch.randn(1, 513, 86, device="cuda", dtype=torch.float32).abs() + 1e-6
        spec_cpu = spec_gpu.detach().cpu()

        gl_gpu = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=32).cuda()
        gl_cpu = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=32)

        torch.cuda.synchronize()
        with torch.amp.autocast('cuda', enabled=False):
            wave_gpu = gl_gpu(spec_gpu)
        torch.cuda.synchronize()
        
        has_nan_gpu = torch.isnan(wave_gpu).any().item()
        sf.write(path_gpu, wave_gpu.detach().cpu().squeeze().numpy(), 22050, format='WAV', subtype='PCM_16')
        
        wave_cpu = gl_cpu(spec_cpu)
        has_nan_cpu = torch.isnan(wave_cpu).any().item()
        sf.write(path_cpu, wave_cpu.squeeze().numpy(), 22050, format='WAV', subtype='PCM_16')

        print("\nOutput Statistics:")
        print(f" - GPU Result: {'FAIL' if has_nan_gpu else 'Pass'}")
        print(f" - CPU Result: {'FAIL' if has_nan_cpu else 'Pass'}")

        print("\n" + "="*40)
        if not has_nan_cpu and has_nan_gpu:
            print("VERDICT: HARDWARE KERNEL INSTABILITY (rocFFT)")
            print("Conclusion: GFX1010 kernels cannot handle this spectral math.")
        elif not has_nan_cpu and not has_nan_gpu:
            print("VERDICT: MATHEMATICAL DOMAIN ERROR")
            print("Conclusion: GPU is healthy. Stability depends on non-negative inputs.")
        
        if not has_nan_gpu:
            print("✅ Forensic audit verified.")
            print("="*40)
            return True
        else:
            print("❌ Forensic audit identified hardware-level instability.")
            print("="*40)
            return False

    except Exception as e:
        print(f"❌ Audit Critical Error: {e}")
        return False