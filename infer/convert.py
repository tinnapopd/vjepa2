import argparse
import re
from pathlib import Path

import torch

from .configuration import VJEPA21Config  # type: ignore
from .modeling import VJEPA21Model  # type: ignore


VARIANT_CONFIGS = {
    "vitb": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "mlp_ratio": 4.0,
        "n_output_distillation": 1,
        "pred_num_hidden_layers": 12,
        "pred_num_mask_tokens": 8,
        "teacher_embed_dim": 1664,
        "checkpoint_key": "ema_encoder",
    },
    "vitl": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "mlp_ratio": 4.0,
        "n_output_distillation": 1,
        "pred_num_hidden_layers": 12,
        "pred_num_mask_tokens": 8,
        "teacher_embed_dim": 1664,
        "checkpoint_key": "ema_encoder",
    },
    "vitg": {
        "hidden_size": 1408,
        "num_hidden_layers": 40,
        "num_attention_heads": 22,
        "mlp_ratio": 48 / 11,
        "n_output_distillation": 4,
        "pred_num_hidden_layers": 24,
        "pred_num_mask_tokens": 8,
        "teacher_embed_dim": None,
        "checkpoint_key": "target_encoder",
    },
    "vitG": {
        "hidden_size": 1664,
        "num_hidden_layers": 48,
        "num_attention_heads": 26,
        "mlp_ratio": 64 / 13,
        "n_output_distillation": 4,
        "pred_num_hidden_layers": 24,
        "pred_num_mask_tokens": 8,
        "teacher_embed_dim": None,
        "checkpoint_key": "target_encoder",
    },
}

PREFIX = "module.backbone."


def strip_prefix(key: str) -> str:
    if key.startswith(PREFIX):
        return key[len(PREFIX) :]
    return key


def convert_encoder_keys(encoder_sd: dict) -> dict:
    """Convert Meta encoder state dict to HF format."""
    new_sd = {}

    for orig_key, tensor in encoder_sd.items():
        key = strip_prefix(orig_key)

        # patch_embed.proj.* -> encoder.embeddings.patch_embeddings.proj.*
        if key.startswith("patch_embed.proj."):
            suffix = key[len("patch_embed.proj.") :]
            new_sd[f"encoder.embeddings.patch_embeddings.proj.{suffix}"] = (
                tensor
            )
            continue

        # patch_embed_img.proj.* -> encoder.embeddings.patch_embeddings_img.proj.*
        if key.startswith("patch_embed_img.proj."):
            suffix = key[len("patch_embed_img.proj.") :]
            new_sd[
                f"encoder.embeddings.patch_embeddings_img.proj.{suffix}"
            ] = tensor
            continue

        # img_mod_embed -> encoder.embeddings.img_mod_embed
        if key == "img_mod_embed":
            new_sd["encoder.embeddings.img_mod_embed"] = tensor
            continue

        # video_mod_embed -> encoder.embeddings.video_mod_embed
        if key == "video_mod_embed":
            new_sd["encoder.embeddings.video_mod_embed"] = tensor
            continue

        # norms_block.N.* -> encoder.norms_block.N.*
        if key.startswith("norms_block."):
            new_sd[f"encoder.{key}"] = tensor
            continue

        # blocks.N.attn.qkv.* -> split into query, key, value
        m = re.match(r"blocks\.(\d+)\.attn\.qkv\.(weight|bias)", key)
        if m:
            layer_idx = m.group(1)
            param_type = m.group(2)
            q, k, v = tensor.chunk(3, dim=0)
            new_sd[
                f"encoder.layer.{layer_idx}.attention.query.{param_type}"
            ] = q
            new_sd[f"encoder.layer.{layer_idx}.attention.key.{param_type}"] = k
            new_sd[
                f"encoder.layer.{layer_idx}.attention.value.{param_type}"
            ] = v
            continue

        # blocks.N.attn.proj.* -> encoder.layer.N.attention.proj.*
        m = re.match(r"blocks\.(\d+)\.attn\.proj\.(.*)", key)
        if m:
            layer_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"encoder.layer.{layer_idx}.attention.proj.{suffix}"] = (
                tensor
            )
            continue

        # blocks.N.norm1.* -> encoder.layer.N.norm1.*
        m = re.match(r"blocks\.(\d+)\.norm1\.(.*)", key)
        if m:
            new_sd[f"encoder.layer.{m.group(1)}.norm1.{m.group(2)}"] = tensor
            continue

        # blocks.N.norm2.* -> encoder.layer.N.norm2.*
        m = re.match(r"blocks\.(\d+)\.norm2\.(.*)", key)
        if m:
            new_sd[f"encoder.layer.{m.group(1)}.norm2.{m.group(2)}"] = tensor
            continue

        # blocks.N.mlp.* -> encoder.layer.N.mlp.*
        m = re.match(r"blocks\.(\d+)\.mlp\.(.*)", key)
        if m:
            new_sd[f"encoder.layer.{m.group(1)}.mlp.{m.group(2)}"] = tensor
            continue

        print(f"WARNING: unmapped encoder key: {orig_key}")

    return new_sd


def convert_predictor_keys(predictor_sd: dict, num_mask_tokens: int) -> dict:
    """Convert Meta predictor state dict to HF format."""
    new_sd = {}
    mask_token_tensors = {}

    for orig_key, tensor in predictor_sd.items():
        key = strip_prefix(orig_key)

        # predictor_embed.* -> predictor.embeddings.predictor_embeddings.*
        if key.startswith("predictor_embed."):
            suffix = key[len("predictor_embed.") :]
            new_sd[f"predictor.embeddings.predictor_embeddings.{suffix}"] = (
                tensor
            )
            continue

        # mask_tokens.N -> collect for stacking
        m = re.match(r"mask_tokens\.(\d+)", key)
        if m:
            idx = int(m.group(1))
            mask_token_tensors[idx] = tensor
            continue

        # img_mod_embed -> predictor.embeddings.img_mod_embed AND predictor.img_mod_embed
        if key == "img_mod_embed":
            new_sd["predictor.embeddings.img_mod_embed"] = tensor.clone()
            new_sd["predictor.img_mod_embed"] = tensor.clone()
            continue

        # video_mod_embed -> predictor.embeddings.video_mod_embed AND predictor.video_mod_embed
        if key == "video_mod_embed":
            new_sd["predictor.embeddings.video_mod_embed"] = tensor.clone()
            new_sd["predictor.video_mod_embed"] = tensor.clone()
            continue

        # predictor_blocks.N.attn.qkv.* -> split into query, key, value
        pm = re.match(
            r"predictor_blocks\.(\d+)\.attn\.qkv\.(weight|bias)", key
        )
        if pm:
            layer_idx = pm.group(1)
            param_type = pm.group(2)
            q, k, v = tensor.chunk(3, dim=0)
            new_sd[
                f"predictor.layer.{layer_idx}.attention.query.{param_type}"
            ] = q
            new_sd[
                f"predictor.layer.{layer_idx}.attention.key.{param_type}"
            ] = k
            new_sd[
                f"predictor.layer.{layer_idx}.attention.value.{param_type}"
            ] = v
            continue

        # predictor_blocks.N.attn.proj.* -> predictor.layer.N.attention.proj.*
        pm = re.match(r"predictor_blocks\.(\d+)\.attn\.proj\.(.*)", key)
        if pm:
            new_sd[
                f"predictor.layer.{pm.group(1)}.attention.proj.{pm.group(2)}"
            ] = tensor
            continue

        # predictor_blocks.N.norm1.* -> predictor.layer.N.norm1.*
        pm = re.match(r"predictor_blocks\.(\d+)\.norm1\.(.*)", key)
        if pm:
            new_sd[f"predictor.layer.{pm.group(1)}.norm1.{pm.group(2)}"] = (
                tensor
            )
            continue

        # predictor_blocks.N.norm2.* -> predictor.layer.N.norm2.*
        pm = re.match(r"predictor_blocks\.(\d+)\.norm2\.(.*)", key)
        if pm:
            new_sd[f"predictor.layer.{pm.group(1)}.norm2.{pm.group(2)}"] = (
                tensor
            )
            continue

        # predictor_blocks.N.mlp.* -> predictor.layer.N.mlp.*
        pm = re.match(r"predictor_blocks\.(\d+)\.mlp\.(.*)", key)
        if pm:
            new_sd[f"predictor.layer.{pm.group(1)}.mlp.{pm.group(2)}"] = tensor
            continue

        # predictor_norm.* -> predictor.layernorm.*
        if key.startswith("predictor_norm."):
            suffix = key[len("predictor_norm.") :]
            new_sd[f"predictor.layernorm.{suffix}"] = tensor
            continue

        # predictor_proj_context.* -> predictor.proj_context.*
        if key.startswith("predictor_proj_context."):
            suffix = key[len("predictor_proj_context.") :]
            new_sd[f"predictor.proj_context.{suffix}"] = tensor
            continue

        # predictor_proj.* -> predictor.proj.* (must come after predictor_proj_context)
        if key.startswith("predictor_proj."):
            suffix = key[len("predictor_proj.") :]
            new_sd[f"predictor.proj.{suffix}"] = tensor
            continue

        print(f"WARNING: unmapped predictor key: {orig_key}")

    # Stack mask tokens: {0: [1,1,D], 1: [1,1,D], ...} -> [N, 1, 1, D]
    if mask_token_tensors:
        stacked = torch.stack(
            [mask_token_tensors[i] for i in range(num_mask_tokens)], dim=0
        )
        new_sd["predictor.embeddings.mask_tokens"] = stacked

    return new_sd


def convert_checkpoint(checkpoint_path: str, variant: str, output_path: str):
    if variant not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant: {variant}. Choose from {list(VARIANT_CONFIGS.keys())}"
        )

    vcfg = VARIANT_CONFIGS[variant]

    # Build HF config
    config = VJEPA21Config(
        hidden_size=vcfg["hidden_size"],
        num_hidden_layers=vcfg["num_hidden_layers"],
        num_attention_heads=vcfg["num_attention_heads"],
        mlp_ratio=vcfg["mlp_ratio"],
        n_output_distillation=vcfg["n_output_distillation"],
        pretrained_crop_size=256,
        pred_num_hidden_layers=vcfg["pred_num_hidden_layers"],
        pred_num_mask_tokens=vcfg["pred_num_mask_tokens"],
        teacher_embed_dim=vcfg["teacher_embed_dim"],
    )

    print(f"Loading checkpoint from {checkpoint_path}...")
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    encoder_key = vcfg["checkpoint_key"]
    print(f"Using encoder key: '{encoder_key}'")
    encoder_sd = sd[encoder_key]
    predictor_sd = sd["predictor"]

    print(f"Converting encoder ({len(encoder_sd)} keys)...")
    hf_encoder = convert_encoder_keys(encoder_sd)

    print(f"Converting predictor ({len(predictor_sd)} keys)...")
    hf_predictor = convert_predictor_keys(
        predictor_sd,
        vcfg["pred_num_mask_tokens"],  # type: ignore
    )

    # Merge
    hf_sd = {}
    hf_sd.update(hf_encoder)
    hf_sd.update(hf_predictor)

    print(f"Total converted keys: {len(hf_sd)}")

    # Create model and load
    print("Creating HF model...")
    model = VJEPA21Model(config)
    model_sd = model.state_dict()

    # Check for mismatches
    converted_keys = set(hf_sd.keys())
    expected_keys = set(model_sd.keys())

    missing = expected_keys - converted_keys
    unexpected = converted_keys - expected_keys

    if missing:
        print(f"\nMISSING keys ({len(missing)}):")
        for k in sorted(missing):
            print(f"  {k}: {model_sd[k].shape}")

    if unexpected:
        print(f"\nUNEXPECTED keys ({len(unexpected)}):")
        for k in sorted(unexpected):
            print(f"  {k}: {hf_sd[k].shape}")

    if missing or unexpected:
        raise RuntimeError(
            f"Key mismatch! {len(missing)} missing, {len(unexpected)} unexpected. Fix the conversion mapping."
        )

    print("All keys match! Loading weights...")
    model.load_state_dict(hf_sd)

    # Save
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output}...")
    model.save_pretrained(output)
    config.save_pretrained(output)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert V-JEPA 2.1 weights to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Meta .pt checkpoint",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=list(VARIANT_CONFIGS.keys()),
        help="Model variant",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory"
    )
    args = parser.parse_args()

    convert_checkpoint(args.checkpoint, args.variant, args.output)


if __name__ == "__main__":
    main()
