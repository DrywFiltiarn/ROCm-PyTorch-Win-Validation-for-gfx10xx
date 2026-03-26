import torch
import torchaudio
import time
import os
import soundfile as sf

def run():
    print(">> Testing Generative Audio Pipeline + Direct Disk I/O...")
    n_fft = 1024
    output_path = os.path.join("logs", "test_output_audio.wav")
    
    print(f"Config: n_fft={n_fft} | Mode: CPU Spectral Bypass | Device: GFX1010")
    
    try:
        spec_gpu = torch.randn(1, 513, 86, device="cuda", dtype=torch.float32).abs() + 1e-6
        
        gpu_nans = torch.isnan(spec_gpu).any().item()
        if gpu_nans:
            print("❌ Critical: NaNs detected on GPU before spectral processing.")
            return False
            
        print(f"Input Stats - Spec Mean: {spec_gpu.mean().item():.4f} | VRAM: {torch.cuda.memory_allocated()/1e6:.2f}MB")

        torch.cuda.synchronize()
        spec_cpu = spec_gpu.detach().cpu()
        
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, 
            n_iter=32, 
            win_length=n_fft, 
            hop_length=n_fft // 4,
            power=2.0
        )

        start = time.time()
        waveform_cpu = griffin_lim(spec_cpu)
        latency = (time.time() - start) * 1000

        cpu_waveform_np = waveform_cpu.squeeze().numpy()
        
        print(f"Saving artifact via Direct Soundfile Backend...")
        sf.write(output_path, cpu_waveform_np, 22050, format='WAV', subtype='PCM_16')
        
        has_nan = torch.isnan(waveform_cpu).any().item()
        max_amp = torch.abs(waveform_cpu).max().item()
        file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

        print("Output Statistics:")
        print(f" - Audio Shape: {waveform_cpu.shape} | Latency: {latency:.2f}ms")
        print(f" - Max Amp:     {max_amp:.4f}")
        print(f" - NaN Check:   {'FAIL' if has_nan else 'Pass'}")
        print(f" - Disk Write:  {'Pass' if file_exists else 'FAIL'}")

        if has_nan or not file_exists or max_amp < 1e-5:
            return False

        print("✅ Audio generation verified (Stable Path).")
        return True

    except Exception as e:
        print(f"❌ Audio Gen Critical Failure: {e}")
        return False