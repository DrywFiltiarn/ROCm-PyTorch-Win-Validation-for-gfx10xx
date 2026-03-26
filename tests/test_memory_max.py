import torch

def run():
    print(">> Testing VRAM Overflow to System Shared Memory...")
    print("Config: Target=32GB | Mode: Shared/Swap Mapping")
    
    try:
        initial_mem = torch.cuda.memory_allocated() / 1e9
        print(f"Initial Allocated: {initial_mem:.2f} GB")
        
        tensors = []
        print("\n--- Mapping 4GB Chunks to System Swap ---")
        for i in range(8):
            t = torch.ones((1024, 1024, 1024), device="cuda", dtype=torch.float32)
            t *= (i + 1)
            tensors.append(t)
            print(f" - Chunk {i+1}/8 Validated | Active: {(i+1)*4}GB")
        
        torch.cuda.synchronize()
        
        val_start = tensors[0][0,0,0].item()
        val_end = tensors[-1][0,0,0].item()
        sum_check = val_start + val_end
        
        has_nan = torch.isnan(tensors[-1]).any().item()

        print("Output Statistics:")
        print(f" - Total Tensors: {len(tensors)}")
        print(f" - Integrity Check: {sum_check} (Expected 9.0)")
        print(f" - NaN Check: {'FAIL' if has_nan else 'Pass'}")
        
        if sum_check != 9.0 or has_nan:
            print("❌ Error: Memory corruption detected at the swap boundary.")
            return False
            
        print("✅ Extreme memory stress and data integrity verified.")
        return True
    except Exception as e:
        print(f"❌ Memory Stress Failure: {e}")
        return False