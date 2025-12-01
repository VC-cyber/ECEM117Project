"""
Quick script to clear GPU memory.
Run this if you get CUDA out of memory errors.
"""

import torch
import gc

print("Clearing GPU memory...")

# Clear PyTorch cache
if torch.cuda.is_available():
    print(f"GPU Memory before: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"GPU Memory after: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
    print("✓ GPU cache cleared!")
else:
    print("No GPU available")

