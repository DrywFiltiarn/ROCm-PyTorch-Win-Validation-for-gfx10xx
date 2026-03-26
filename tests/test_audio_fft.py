import torch
import time

def run():
    print(">> Testing Signal Processing (rocFFT)...")
    length = 44100
    print(f"Config: Len={length} | Mode: Complex64")
    
    try:
        t = torch.linspace(0, 1, length, device="cuda")
        signal = torch.sin(2 * 3.14159 * 440 * t) # 440Hz Sine
        
        print("Executing Forward & Inverse FFT...")
        torch.cuda.synchronize()
        start = time.time()
        spectrum = torch.fft.fft(signal)
        reconstructed = torch.fft.ifft(spectrum).real
        torch.cuda.synchronize()
        
        error = torch.abs(signal - reconstructed).max().item()
        has_nan = torch.isnan(spectrum).any().item()

        print(f"Output Statistics:")
        print(f" - Latency: {(time.time()-start)*1000:.2f}ms")
        print(f" - Max Reconstruction Delta: {error:.2e}")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
        
        if has_nan or error > 1e-4:
            print("❌ Error: FFT precision failure or NaN detected.")
            return False
            
        print("✅ FFT kernels verified.")
        return True
    except Exception as e:
        print(f"❌ rocFFT Failure: {e}")
        return False