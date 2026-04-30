#!/usr/bin/env python3
"""
Clip-level evaluation of V-JEPA 2.1 on weapon detection dataset.

Usage:
    python inference.py --variant vitl --dataset /path/to/dataset
    python inference.py --variant vitg --dataset /path/to/dataset
    python inference.py --variant vitl --dataset /path/to/dataset --device cpu
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import src.models.vision_transformer as vit  # type: ignore
from src.models.attentive_pooler import AttentiveClassifier  # type: ignore

# ---------------------------------------------------------------------------
# Variant configs
# ---------------------------------------------------------------------------
VARIANT_CONFIGS = {
    "vitl": {
        "model_name": "vit_large",
        "embed_dim": 1024,
        "checkpoint_key": "ema_encoder",
        "encoder_path": "pretrained-models/vjepa2_1_vitl_dist_vitG_384.pt",
        "probe_path": "trained-probes/vitl-probe.pt",
        "num_heads": 16,
    },
    "vitg": {
        "model_name": "vit_giant_xformers",
        "embed_dim": 1408,
        "checkpoint_key": "target_encoder",
        "encoder_path": "pretrained-models/vjepa2_1_vitG_384.pt",
        "probe_path": "trained-probes/vitG-probe.pt",
        "num_heads": 22,
    },
}

LABELS = ["background", "weaponized"]


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------
def load_video_clips(video_path, clip_frames=64):
    """Load video and split into non-overlapping clips of `clip_frames` frames.

    Returns list of (clip_tensor, start_sec, end_sec).
    clip_tensor shape: (C, F, H, W) float32 normalized.
    """
    try:
        import decord  # type: ignore

        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(video_path), num_threads=1)
        total_frames = len(vr)
        fps = vr.get_avg_fps()

        clips = []
        for start in range(0, total_frames, clip_frames):
            end = min(start + clip_frames, total_frames)
            if end - start < clip_frames // 2:
                break  # skip if too short

            indices = list(range(start, end))
            # If fewer than clip_frames, uniformly resample to clip_frames
            if len(indices) < clip_frames:
                indices = (
                    torch.linspace(start, end - 1, steps=clip_frames)
                    .round()
                    .long()
                    .tolist()
                )

            frames = vr.get_batch(indices)  # (F, H, W, C) uint8 torch
            # Convert to list of numpy arrays for transform compatibility
            frames_np = [frames[i].numpy() for i in range(frames.shape[0])]

            start_sec = start / fps
            end_sec = end / fps
            clips.append((frames_np, start_sec, end_sec))

        return clips, fps, total_frames

    except ImportError:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        clips = []
        for start in range(0, total_frames, clip_frames):
            end = min(start + clip_frames, total_frames)
            if end - start < clip_frames // 2:
                break

            indices = list(range(start, end))
            if len(indices) < clip_frames:
                indices = (
                    torch.linspace(start, end - 1, steps=clip_frames)
                    .round()
                    .long()
                    .tolist()
                )

            frames_np = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_np.append(frame)
                else:
                    frames_np.append(np.zeros((1, 1, 3), dtype=np.uint8))

            start_sec = start / fps
            end_sec = end / fps
            clips.append((frames_np, start_sec, end_sec))

        cap.release()
        return clips, fps, total_frames


def preprocess_clip(frames_np, crop_size=384):
    """Preprocess a list of numpy frames (H, W, C) uint8 to (C, F, H, W) tensor.

    Uses the same resize/crop/normalize as evals/ VideoTransform (eval mode).
    """
    import src.datasets.utils.video.transforms as video_transforms  # type: ignore
    import src.datasets.utils.video.volume_transforms as volume_transforms  # type: ignore

    transform = video_transforms.Compose(
        [
            video_transforms.Resize(
                int(crop_size * 256 / 224), interpolation="bilinear"
            ),
            video_transforms.CenterCrop(size=(crop_size, crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    clip_tensor = transform(frames_np)  # (C, F, H, W)
    return clip_tensor


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------
def load_violent_intervals(label_path):
    """Load violent time intervals from a CSV file.

    Each row: start_sec, end_sec
    Returns list of (start, end) tuples.
    """
    intervals = []
    if not os.path.exists(label_path):
        return intervals
    try:
        with open(label_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        start = float(row[0].strip())
                        end = float(row[1].strip())
                        intervals.append((start, end))
                    except ValueError:
                        continue
    except Exception:
        pass
    return intervals


def clip_overlaps_interval(start_sec, end_sec, intervals):
    """Check if clip time range overlaps with any violent interval."""
    for iv_start, iv_end in intervals:
        if start_sec < iv_end and end_sec > iv_start:
            return True
    return False


def scan_dataset(dataset_root):
    """Scan the dataset directory.

    Returns list of dicts:
        {
            "video_path": str,
            "label_path": str or None,
            "category": "violent" | "non-violent",
            "intervals": [(start, end), ...],
        }
    """
    dataset_root = Path(dataset_root)
    samples = []
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    for category in ["violent", "non-violent"]:
        video_dir = dataset_root / category / "videos"
        label_dir = dataset_root / category / "labels"

        if not video_dir.exists():
            print(f"WARNING: {video_dir} not found, skipping")
            continue

        for vf in sorted(video_dir.iterdir()):
            if vf.suffix.lower() not in video_exts:
                continue

            label_path = (
                label_dir / (vf.stem + ".csv") if label_dir.exists() else None
            )
            intervals = []
            if category == "violent" and label_path and label_path.exists():
                intervals = load_violent_intervals(str(label_path))

            samples.append(
                {
                    "video_path": str(vf),
                    "label_path": str(label_path) if label_path else None,
                    "category": category,
                    "intervals": intervals,
                }
            )

    return samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_encoder(variant_cfg, device, resolution=384, frames_per_clip=16):
    """Load the frozen encoder using Meta's native checkpoint format."""
    encoder_path = variant_cfg["encoder_path"]
    model_name = variant_cfg["model_name"]
    checkpoint_key = variant_cfg["checkpoint_key"]

    print(f"Loading encoder checkpoint: {encoder_path}")
    checkpoint = torch.load(
        encoder_path, map_location="cpu", weights_only=True
    )

    # Build encoder
    encoder = vit.__dict__[model_name](
        img_size=resolution,
        num_frames=frames_per_clip,
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )

    # Load weights
    pretrained_dict = checkpoint[checkpoint_key]
    pretrained_dict = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in pretrained_dict.items()
    }

    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            print(f"  WARNING: key '{k}' not in checkpoint")
        elif pretrained_dict[k].shape != v.shape:
            print(
                f"  WARNING: shape mismatch for '{k}': {pretrained_dict[k].shape} vs {v.shape}"
            )
            pretrained_dict[k] = v

    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(f"  Encoder loaded: {msg}")

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    del checkpoint
    return encoder


def load_classifier(variant_cfg, device, num_classes=2, num_probe_blocks=4):
    """Load the trained AttentiveClassifier probe."""
    probe_path = variant_cfg["probe_path"]
    embed_dim = variant_cfg["embed_dim"]
    num_heads = variant_cfg["num_heads"]

    print(f"Loading probe checkpoint: {probe_path}")
    probe_ckpt = torch.load(probe_path, map_location="cpu", weights_only=True)

    classifier = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        depth=num_probe_blocks,
        num_classes=num_classes,
        use_activation_checkpointing=False,
    )

    # The probe checkpoint stores classifiers as a list (from DDP training)
    probe_sd = probe_ckpt["classifiers"][0]
    # Strip "module." prefix from DDP
    probe_sd = {k.replace("module.", ""): v for k, v in probe_sd.items()}

    msg = classifier.load_state_dict(probe_sd, strict=True)
    print(f"  Classifier loaded: {msg}")

    classifier = classifier.to(device).eval()
    for p in classifier.parameters():
        p.requires_grad = False

    del probe_ckpt
    return classifier


def auto_detect_variant():
    """Auto-detect which variant is available based on file existence."""
    for name, cfg in VARIANT_CONFIGS.items():
        if os.path.exists(cfg["encoder_path"]) and os.path.exists(
            cfg["probe_path"]
        ):
            print(f"Auto-detected variant: {name}")
            return name
    raise FileNotFoundError(
        "No matching encoder+probe pair found. Available configs:\n"
        + "\n".join(
            f"  {k}: encoder={v['encoder_path']}, probe={v['probe_path']}"
            for k, v in VARIANT_CONFIGS.items()
        )
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_clip(encoder, classifier, clip_tensor, device, use_bfloat16=True):
    """Run inference on a single clip.

    Args:
        clip_tensor: (C, F, H, W) preprocessed tensor
    Returns:
        (predicted_label_idx, probability_dict)
    """
    # Add batch dim: (1, C, F, H, W)
    x = clip_tensor.unsqueeze(0).to(device)

    with torch.cuda.amp.autocast(
        dtype=torch.float16, enabled=use_bfloat16 and device.type == "cuda"
    ):
        features = encoder(x)  # (1, num_tokens, embed_dim)
        logits = classifier(features)  # (1, num_classes)

    probs = F.softmax(logits, dim=-1)[0].cpu().float()
    pred_idx = probs.argmax().item()
    prob_dict = {LABELS[i]: probs[i].item() for i in range(len(LABELS))}

    return pred_idx, prob_dict


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(tp, tn, fp, fn):
    """Compute accuracy, precision, recall, F1 from confusion matrix counts."""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def print_video_stats(
    video_idx,
    total_videos,
    video_name,
    category,
    video_tp,
    video_tn,
    video_fp,
    video_fn,
    num_clips,
    elapsed,
):
    """Print per-video statistics after all clips are processed."""
    m = compute_metrics(video_tp, video_tn, video_fp, video_fn)
    print(f"\n{'=' * 80}")
    print(f"[{video_idx + 1:3d}/{total_videos}] {category}/{video_name}")
    print(f"  Clips: {num_clips}  |  Time: {elapsed:.1f}s")
    print(f"  TP: {video_tp}  TN: {video_tn}  FP: {video_fp}  FN: {video_fn}")
    print(
        f"  Acc: {m['accuracy']:.4f}  Prec: {m['precision']:.4f}  "
        f"Rec: {m['recall']:.4f}  F1: {m['f1']:.4f}"
    )
    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# Summary saving
# ---------------------------------------------------------------------------
def save_summary(all_results, global_metrics, output_path, args_info):
    """Save evaluation report to file."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("V-JEPA 2.1 Clip-Level Evaluation Summary\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Variant: {args_info['variant']}\n")
        f.write(f"Dataset: {args_info['dataset']}\n")
        f.write(f"Device: {args_info['device']}\n")
        f.write(f"Clip frames: {args_info['clip_frames']}\n")
        f.write(f"Resolution: {args_info['resolution']}\n")
        f.write("=" * 80 + "\n\n")

        f.write("--- Overall Metrics (clip-level) ---\n")
        f.write(f"Total clips: {global_metrics['total']}\n")
        f.write(f"Accuracy:    {global_metrics['accuracy']:.4f}\n")
        f.write(f"Precision:   {global_metrics['precision']:.4f}\n")
        f.write(f"Recall:      {global_metrics['recall']:.4f}\n")
        f.write(f"F1:          {global_metrics['f1']:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(
            f"  TP: {global_metrics['tp']:5d}    FP: {global_metrics['fp']:5d}\n"
        )
        f.write(
            f"  FN: {global_metrics['fn']:5d}    TN: {global_metrics['tn']:5d}\n\n"
        )

        f.write("-" * 80 + "\n")
        f.write("Per-Video Details\n")
        f.write("-" * 80 + "\n")
        header = f"{'Video':<50} {'Cat':<12} {'Clips':>5} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'Acc':>6} {'F1':>6}\n"
        f.write(header)
        f.write("-" * 80 + "\n")

        for r in all_results:
            m = compute_metrics(r["tp"], r["tn"], r["fp"], r["fn"])
            name = Path(r["video_path"]).name
            if len(name) > 48:
                name = name[:45] + "..."
            f.write(
                f"{name:<50} {r['category']:<12} {r['num_clips']:>5} "
                f"{r['tp']:>4} {r['tn']:>4} {r['fp']:>4} {r['fn']:>4} "
                f"{m['accuracy']:>6.3f} {m['f1']:>6.3f}\n"
            )

        f.write("-" * 80 + "\n")

    # Also save as JSON
    json_path = output_path.replace(".txt", ".json")
    json_data = {
        "metadata": args_info,
        "global_metrics": global_metrics,
        "per_video": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print(f"JSON saved to:    {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="V-JEPA 2.1 Clip-Level Evaluation"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=list(VARIANT_CONFIGS.keys()),
        help="Model variant (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset root (with violent/ and non-violent/ subdirs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=16,
        help="Number of frames per clip (default: 16)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=384,
        help="Input resolution (default: 384)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.txt",
        help="Output file path",
    )
    parser.add_argument(
        "--num-probe-blocks",
        type=int,
        default=4,
        help="Number of probe self-attention blocks (default: 4)",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 mixed precision",
    )
    args = parser.parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Variant
    if args.variant is None:
        variant_name = auto_detect_variant()
    else:
        variant_name = args.variant
    variant_cfg = VARIANT_CONFIGS[variant_name]

    # Load models
    print("\n--- Loading Models ---")
    encoder = load_encoder(
        variant_cfg,
        device,
        resolution=args.resolution,
        frames_per_clip=args.clip_frames,
    )
    classifier = load_classifier(
        variant_cfg,
        device,
        num_classes=len(LABELS),
        num_probe_blocks=args.num_probe_blocks,
    )
    print("Models loaded successfully.\n")

    # Scan dataset
    print("--- Scanning Dataset ---")
    samples = scan_dataset(args.dataset)
    n_violent = sum(1 for s in samples if s["category"] == "violent")
    n_nonviolent = sum(1 for s in samples if s["category"] == "non-violent")
    print(
        f"Found {len(samples)} videos: {n_violent} violent, {n_nonviolent} non-violent\n"
    )

    if not samples:
        print("ERROR: No videos found in dataset.")
        sys.exit(1)

    # Evaluate
    print("--- Running Evaluation ---")
    use_bf16 = not args.no_bf16

    global_tp, global_tn, global_fp, global_fn = 0, 0, 0, 0
    all_results = []
    total_start = time.time()

    for vid_idx, sample in enumerate(samples):
        video_path = sample["video_path"]
        category = sample["category"]
        intervals = sample["intervals"]
        video_name = Path(str(video_path)).name

        vid_start = time.time()

        try:
            clips, fps, total_frames = load_video_clips(
                video_path,
                clip_frames=args.clip_frames,
            )
        except Exception as e:
            print(f"\n  ERROR loading {video_name}: {e}")
            all_results.append(
                {
                    "video_path": video_path,
                    "category": category,
                    "num_clips": 0,
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "error": str(e),
                }
            )
            continue

        video_tp, video_tn, video_fp, video_fn = 0, 0, 0, 0

        for clip_idx, (frames_np, start_sec, end_sec) in enumerate(clips):
            # Determine ground truth for this clip
            if category == "violent":
                gt_positive = clip_overlaps_interval(
                    start_sec, end_sec, intervals
                )
            else:
                gt_positive = False

            # Preprocess and predict
            try:
                clip_tensor = preprocess_clip(
                    frames_np, crop_size=args.resolution
                )
                pred_idx, prob_dict = predict_clip(
                    encoder,
                    classifier,
                    clip_tensor,
                    device,
                    use_bfloat16=use_bf16,
                )
            except Exception as e:
                print(f"  ERROR on clip {clip_idx} of {video_name}: {e}")
                continue

            pred_positive = pred_idx == 1  # class 1 = fight/violent

            # Update confusion matrix
            if gt_positive and pred_positive:
                video_tp += 1
            elif not gt_positive and not pred_positive:
                video_tn += 1
            elif not gt_positive and pred_positive:
                video_fp += 1
            else:
                video_fn += 1

            # Print per-clip inline
            gt_str = "POS" if gt_positive else "NEG"
            pred_str = "POS" if pred_positive else "NEG"
            match = "✓" if gt_positive == pred_positive else "✗"
            fight_prob = prob_dict.get("fight", 0)
            print(
                f"  clip {clip_idx + 1:3d}/{len(clips)} "
                f"[{start_sec:6.1f}s-{end_sec:6.1f}s] "
                f"GT:{gt_str} Pred:{pred_str} "
                f"fight={fight_prob:.3f} {match}",
                end="\r",
            )

        elapsed = time.time() - vid_start

        # Accumulate global
        global_tp += video_tp
        global_tn += video_tn
        global_fp += video_fp
        global_fn += video_fn

        # Print per-video summary
        print_video_stats(
            vid_idx,
            len(samples),
            video_name,
            category,
            video_tp,
            video_tn,
            video_fp,
            video_fn,
            len(clips),
            elapsed,
        )

        # Running global metrics
        gm = compute_metrics(global_tp, global_tn, global_fp, global_fn)
        print(
            f"  Running Global → Acc: {gm['accuracy']:.4f}  "
            f"Prec: {gm['precision']:.4f}  Rec: {gm['recall']:.4f}  "
            f"F1: {gm['f1']:.4f}  "
            f"(TP:{global_tp} TN:{global_tn} FP:{global_fp} FN:{global_fn})"
        )

        all_results.append(
            {
                "video_path": video_path,
                "category": category,
                "num_clips": len(clips),
                "tp": video_tp,
                "tn": video_tn,
                "fp": video_fp,
                "fn": video_fn,
            }
        )

    # Final summary
    total_elapsed = time.time() - total_start
    global_metrics = compute_metrics(
        global_tp, global_tn, global_fp, global_fn
    )

    print(f"\n{'#' * 80}")
    print("FINAL RESULTS")
    print(f"{'#' * 80}")
    print(f"Total clips evaluated: {global_metrics['total']}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Accuracy:  {global_metrics['accuracy']:.4f}")
    print(f"Precision: {global_metrics['precision']:.4f}")
    print(f"Recall:    {global_metrics['recall']:.4f}")
    print(f"F1:        {global_metrics['f1']:.4f}")
    print(
        f"TP: {global_tp}  TN: {global_tn}  FP: {global_fp}  FN: {global_fn}"
    )
    print(f"{'#' * 80}")

    # Save
    args_info = {
        "variant": variant_name,
        "dataset": args.dataset,
        "device": str(device),
        "clip_frames": args.clip_frames,
        "resolution": args.resolution,
        "timestamp": datetime.now().isoformat(),
        "total_time_sec": total_elapsed,
    }
    save_summary(all_results, global_metrics, args.output, args_info)


if __name__ == "__main__":
    main()
