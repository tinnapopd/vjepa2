#!/bin/bash
cat << 'EOF' > check_models.py
import torch
import glob
import os
import re

def analyze_checkpoint(path):
    print(f"====== File: {path} ======")
    file_size = os.path.getsize(path) / (1024**3)
    print(f"File size: {file_size:.2f} GB")
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        if "ema_encoder" in ckpt:
            state_dict = ckpt["ema_encoder"]
            print("Found: ema_encoder")
        elif "target_encoder" in ckpt:
            state_dict = ckpt["target_encoder"]
            print("Found: target_encoder")
        elif "encoder" in ckpt:
            state_dict = ckpt["encoder"]
            print("Found: encoder")
        elif "classifiers" in ckpt:
            state_dict = ckpt["classifiers"][0]
            print("Found: classifiers")
        else:
            state_dict = ckpt
            print("Found: direct state_dict")
            
        # Normalize keys for easier matching
        norm_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        # Check embedding size
        embed_dim = "Unknown"
        if 'patch_embed.proj.weight' in norm_dict:
            embed_shape = norm_dict['patch_embed.proj.weight'].shape
            print(f"Patch Embed Weight Shape: {embed_shape}")
            embed_dim = embed_shape[0]
        elif 'norm.weight' in norm_dict:
            embed_dim = norm_dict['norm.weight'].shape[0]
            print(f"Norm Weight Shape: {norm_dict['norm.weight'].shape}")
        elif 'pool.weight' in norm_dict:
            embed_dim = norm_dict['pool.weight'].shape[1]
            print(f"Pool Weight Shape: {norm_dict['pool.weight'].shape}")
        
        print(f"--> Estimated Embedding Dimension: {embed_dim}")

        # Check Depth (blocks)
        blocks = set()
        for k in norm_dict.keys():
            match = re.search(r'blocks\.(\d+)', k)
            if match:
                blocks.add(int(match.group(1)))
            match2 = re.search(r'layers\.(\d+)', k)
            if match2:
                blocks.add(int(match2.group(1)))
                
        if blocks:
            print(f"--> Number of Blocks/Layers: {max(blocks) + 1}")
        
        # Look at the final classification heads or probe components
        for k in norm_dict.keys():
            if 'head' in k and 'weight' in k:
                print(f"Head {k} shape: {norm_dict[k].shape}")
            elif 'proj.weight' in k and 'patch' not in k:
                print(f"Proj {k} shape: {norm_dict[k].shape}")
                
    except Exception as e:
        print(f"Error loading {path}: {e}")
    print()

def main():
    models = sorted(glob.glob("pretrained-models/*.pt") + glob.glob("trained-probes/*.pt"))
    if not models:
        print("No .pt models found in pretrained-models/ or trained-probes/ directories.")
    for m in models:
        analyze_checkpoint(m)

if __name__ == "__main__":
    main()
EOF

echo "Running check_models.py..."
python3 check_models.py
