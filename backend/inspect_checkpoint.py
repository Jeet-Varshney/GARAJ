"""
Checkpoint Inspection Script for LA_model.pth (Phase 1C).
Inspects checkpoint file size, state_dict keys, parameter shapes, and layer mappings.
"""

import os
import sys
import torch

checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints", "LA_model.pth"))

def inspect():
    print("=" * 65)
    print("Programmatic Checkpoint Inspection — LA_model.pth")
    print("=" * 65)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at: '{checkpoint_path}'")
        return

    size_bytes = os.path.getsize(checkpoint_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Checkpoint Path : {checkpoint_path}")
    print(f"File Size       : {size_bytes:,} bytes ({size_mb:.2f} MB)")

    print("\nLoading PyTorch checkpoint file with torch.load()...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        print("Root Checkpoint Format: Dictionary containing 'state_dict' key.")
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        print("Root Checkpoint Format: PyTorch State Dictionary.")
        state_dict = ckpt
    else:
        print(f"Root Checkpoint Format: Direct Model Object ({type(ckpt)})")
        state_dict = ckpt.state_dict() if hasattr(ckpt, "state_dict") else {}

    all_keys = list(state_dict.keys())
    print(f"Total Tensor Keys: {len(all_keys)}")

    # Categorize keys by prefix
    prefixes = set()
    ssl_keys = []
    backend_keys = []
    output_keys = []

    total_params = 0
    for k, tensor in state_dict.items():
        if hasattr(tensor, "numel"):
            total_params += tensor.numel()
        prefix = k.split(".")[0]
        prefixes.add(prefix)

        if "ssl" in k.lower() or "wav2vec" in k.lower() or "w2v" in k.lower() or "encoder" in k.lower():
            ssl_keys.append(k)
        else:
            backend_keys.append(k)

        if "out" in k.lower() or "classif" in k.lower() or "fc" in k.lower() or "linear" in k.lower() or "proj" in k.lower():
            output_keys.append((k, getattr(tensor, "shape", None)))

    print(f"Top-Level Key Prefixes: {sorted(list(prefixes))}")
    print(f"SSL Front-End Keys Count : {len(ssl_keys)}")
    print(f"AASIST Back-End Keys Count : {len(backend_keys)}")
    print(f"Total Parameters Count   : {total_params:,}")

    print("\nClassification / Output Layers Found:")
    for name, shape in output_keys[-10:]:
        print(f"  - {name}: shape {shape}")

    print("\nFirst 15 State Dict Keys Sample:")
    for k in all_keys[:15]:
        shape = tuple(state_dict[k].shape) if hasattr(state_dict[k], "shape") else "N/A"
        print(f"  • {k} -> shape {shape}")

    has_ssl = len(ssl_keys) > 0
    has_aasist = len(backend_keys) > 0
    print("\nComposition Summary:")
    print(f"  - XLS-R / SSL Front-End Weights Present : {has_ssl}")
    print(f"  - AASIST Back-End Weights Present       : {has_aasist}")
    print(f"  - Complete End-to-End Weights Present   : {has_ssl and has_aasist}")

if __name__ == "__main__":
    inspect()
