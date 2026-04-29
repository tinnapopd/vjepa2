#!/usr/bin/env python3
"""
V-JEPA 2.1 Inference Script for Weapon Detection Evaluation
============================================================

Loads a pretrained V-JEPA 2.1 encoder and a trained AttentiveClassifier
probe, runs inference on every temporal clip in the evaluation dataset,
and reports classification metrics.

Usage:
    python evaluate_vjepa2_1.py \
        --dataset-dir /path/to/dataset \
        --encoder-ckpt pretrained-models/vjepa2_1_vitb_dist_vitG_384.pt \
        --probe-ckpt /path/to/trained_probe/latest.pt \
        --output-dir ./output/evaluation_results
"""

import argparse
import csv
import logging
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu  # type: ignore

import src.datasets.utils.video.transforms as video_transforms  # type: ignore
import src.datasets.utils.video.volume_transforms as volume_transforms  # type: ignore
import app.vjepa_2_1.models.vision_transformer as vit_v21  # type: ignore
import src.models.vision_transformer as vit_v1  # type: ignore
from src.models.attentive_pooler import AttentiveClassifier  # type: ignore

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = {0: "non-violent", 1: "violent"}


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="V-JEPA 2.1 Inference — Weapon Detection Evaluation"
    )
    # Data
    p.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root of the evaluation dataset",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./output/evaluation_results",
        help="Where to save results CSV and metrics",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit number of videos (for debugging)",
    )

    # Model
    p.add_argument(
        "--encoder-ckpt",
        type=str,
        default="pretrained-models/vjepa2_1_vitb_dist_vitG_384.pt",
        help="Path to V-JEPA 2.1 encoder checkpoint",
    )
    p.add_argument(
        "--probe-ckpt",
        type=str,
        required=True,
        help="Path to trained attentive probe (latest.pt)",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="auto",
        help="Encoder architecture name, or 'auto' to detect from checkpoint "
        "(e.g. vit_base, vit_large, vit_giant_xformers, vit_gigantic_xformers)",
    )
    p.add_argument(
        "--checkpoint-key",
        type=str,
        default="target_encoder",
        help="Key to extract encoder weights from checkpoint",
    )
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--frames-per-clip", type=int, default=16)
    p.add_argument("--frame-step", type=int, default=4)
    p.add_argument("--tubelet-size", type=int, default=2)

    # Classifier
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--probe-depth", type=int, default=4)

    # Inference
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.0,
        help="Min overlap (sec) to label a clip as violent",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Label parsing
# ──────────────────────────────────────────────────────────────────────
def parse_temporal_labels(csv_path: str) -> list[tuple[float, float]]:
    """Read temporal annotation CSV → list of (start_sec, end_sec)."""
    intervals: list[tuple[float, float]] = []
    if not os.path.exists(csv_path):
        return intervals
    with open(csv_path, "r") as f:
        content = f.read().strip()
    if not content:
        return intervals
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                intervals.append(
                    (float(parts[0].strip()), float(parts[1].strip()))
                )
            except ValueError:
                continue
    return intervals


def clip_is_violent(
    clip_start: float,
    clip_end: float,
    intervals: list[tuple[float, float]],
    threshold: float = 0.0,
) -> bool:
    """True if clip time-range overlaps any annotated violent interval."""
    for s, e in intervals:
        overlap = max(0.0, min(clip_end, e) - max(clip_start, s))
        if overlap > threshold:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────
#  Dataset discovery
# ──────────────────────────────────────────────────────────────────────
def discover_dataset(dataset_dir: str) -> list[dict]:
    """Return list of {video_path, label_path, class_label, class_name, video_name}."""
    samples = []
    for class_name, class_label in [("violent", 1), ("non-violent", 0)]:
        videos_dir = os.path.join(dataset_dir, class_name, "videos")
        labels_dir = os.path.join(dataset_dir, class_name, "labels")
        if not os.path.isdir(videos_dir):
            logger.warning(f"Videos directory not found: {videos_dir}")
            continue
        for fname in sorted(os.listdir(videos_dir)):
            if not fname.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                continue
            stem = os.path.splitext(fname)[0]
            samples.append(
                {
                    "video_path": os.path.join(videos_dir, fname),
                    "label_path": os.path.join(labels_dir, f"{stem}.csv"),
                    "class_label": class_label,
                    "class_name": class_name,
                    "video_name": fname,
                }
            )
    return samples


# ──────────────────────────────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────────────────────────────
def build_eval_transform(resolution: int):
    """Eval-time transform: resize → center-crop → to-tensor → normalize."""
    short_side = int(256.0 / 224 * resolution)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(resolution, resolution)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ──────────────────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────────────────
def _detect_model_from_checkpoint(state: dict) -> tuple[str, int, int]:
    """Detect ViT architecture from checkpoint state dict.

    Returns (model_name, embed_dim, depth) by inspecting weight shapes.
    """
    # Detect embed_dim from first block's norm1.weight
    embed_dim = None
    for k, v in state.items():
        if "blocks.0.norm1.weight" in k:
            embed_dim = v.shape[0]
            break

    # Detect depth by counting unique block indices
    block_indices = set()
    for k in state.keys():
        if "blocks." in k:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "blocks" and i + 1 < len(parts):
                    try:
                        block_indices.add(int(parts[i + 1]))
                    except ValueError:
                        continue
    depth = len(block_indices)

    # Map (embed_dim, depth) → model constructor name
    KNOWN_ARCHS = {
        (768, 12): "vit_base",
        (1024, 24): "vit_large",
        (1280, 32): "vit_huge",
        (1408, 40): "vit_giant_xformers",
        (1664, 48): "vit_gigantic_xformers",
    }
    model_name = KNOWN_ARCHS.get((embed_dim, depth))

    return model_name, embed_dim, depth


def _detect_probe_embed_dim(probe_ckpt_path: str) -> int | None:
    """Peek at probe checkpoint to determine its embed_dim."""
    try:
        ckpt = torch.load(probe_ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt["classifiers"][0]
        for k, v in state.items():
            # query_tokens shape is [1, 1, embed_dim]
            if "query_tokens" in k:
                return v.shape[-1]
            # fallback: first norm weight
            if "norm1.weight" in k:
                return v.shape[0]
    except Exception as e:
        logger.warning(f"Could not detect probe embed_dim: {e}")
    return None


def load_encoder(
    ckpt_path, model_name, checkpoint_key, resolution, frames_per_clip,
    device, probe_embed_dim=None,
):
    """Load V-JEPA encoder and freeze.

    Uses the eval-pipeline architecture from src.models.vision_transformer
    (the same module used by evals/video_classification_frozen/modelcustom/
    vit_encoder_multiclip.py). This ensures the encoder embed_dim matches
    the probe that was trained against it.

    If --model-name is 'auto', the architecture is detected from the
    checkpoint weight shapes.
    """
    logger.info(f"Loading encoder from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # --- Resolve checkpoint key ---
    if checkpoint_key in ckpt:
        state = ckpt[checkpoint_key]
    else:
        logger.warning(f"Key '{checkpoint_key}' not found in checkpoint.")
        fallbacks = ["target_encoder", "ema_encoder", "encoder", "model"]
        state = None
        for k in fallbacks:
            if k in ckpt:
                logger.info(f"Using fallback key: '{k}'")
                state = ckpt[k]
                break
        if state is None:
            raise KeyError(
                f"No encoder weights found in {ckpt_path}. "
                f"Tried '{checkpoint_key}' and {fallbacks}. "
                f"Available keys: {list(ckpt.keys())}"
            )

    # Strip DDP / wrapper prefixes (module. from DDP, backbone. from MultiSeqWrapper)
    state = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in state.items()
    }

    # --- Auto-detect model architecture ---
    detected_name, ckpt_embed_dim, ckpt_depth = _detect_model_from_checkpoint(
        state
    )

    # Choose which ViT module to use:
    # - The eval pipeline (vit_encoder_multiclip.py) uses src.models.vision_transformer
    #   so the probe was trained with that module's architecture.
    # - If the probe embed_dim matches a model in src.models, use src.models.
    # - Otherwise fall back to app.vjepa_2_1.models (full V-JEPA 2.1 architecture).
    vit_module = vit_v1  # default: same as eval pipeline
    vit_source = "src.models"

    if model_name == "auto":
        # If probe embed_dim is available, find the model that matches it
        if probe_embed_dim is not None:
            # Find which model name has this embed_dim in src.models
            for name, edim in vit_v1.VIT_EMBED_DIMS.items():
                if edim == probe_embed_dim and name in vit_v1.__dict__:
                    model_name = name
                    logger.info(
                        f"Selected model '{model_name}' to match probe "
                        f"embed_dim={probe_embed_dim}"
                    )
                    break
        if model_name == "auto":
            # Fall back to auto-detect from checkpoint
            if detected_name is None:
                raise ValueError(
                    f"Cannot auto-detect model from checkpoint "
                    f"(embed_dim={ckpt_embed_dim}, depth={ckpt_depth})."
                )
            model_name = detected_name
            logger.info(
                f"Auto-detected model: {model_name} "
                f"(embed_dim={ckpt_embed_dim}, depth={ckpt_depth})"
            )
    else:
        if detected_name and detected_name != model_name:
            logger.warning(
                f"Checkpoint: embed_dim={ckpt_embed_dim}, depth={ckpt_depth} "
                f"→ expected '{detected_name}', but --model-name='{model_name}'."
            )

    # Try src.models first, then app.vjepa_2_1.models
    if model_name in vit_v1.__dict__:
        vit_module = vit_v1
        vit_source = "src.models"
    elif model_name in vit_v21.__dict__:
        vit_module = vit_v21
        vit_source = "app.vjepa_2_1.models"
    else:
        available = list(vit_v1.VIT_EMBED_DIMS.keys())
        raise ValueError(
            f"Model '{model_name}' not found. Available: {available}"
        )

    logger.info(f"Using ViT from {vit_source}.vision_transformer")

    # Build model with the same kwargs used by the official eval configs
    model = vit_module.__dict__[model_name](
        img_size=resolution,
        num_frames=frames_per_clip,
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )

    # Handle shape mismatches
    n_mismatched = 0
    for k, v in model.state_dict().items():
        if k not in state:
            continue
        if state[k].shape != v.shape:
            logger.warning(
                f"Shape mismatch for '{k}': "
                f"ckpt={list(state[k].shape)} vs model={list(v.shape)}"
            )
            state[k] = v
            n_mismatched += 1

    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys:
        logger.warning(f"Missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        logger.info(f"Unexpected keys (ignored): {len(msg.unexpected_keys)}")
    logger.info(
        f"Encoder loaded (embed_dim={model.embed_dim}, "
        f"shape mismatches={n_mismatched})"
    )

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    del ckpt
    return model


def load_classifier(
    ckpt_path, embed_dim, num_heads, depth, num_classes, device
):
    """Load trained AttentiveClassifier probe and freeze.

    embed_dim should match the encoder that was used during probe training.
    """
    logger.info(f"Loading classifier probe from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # First classifier head (index 0)
    state = ckpt["classifiers"][0]
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Auto-detect embed_dim from probe weights if it differs
    probe_embed_dim = None
    for k, v in state.items():
        if "query_tokens" in k:
            probe_embed_dim = v.shape[-1]
            break
    if probe_embed_dim and probe_embed_dim != embed_dim:
        logger.warning(
            f"Probe embed_dim={probe_embed_dim} differs from encoder "
            f"embed_dim={embed_dim}. Using probe embed_dim."
        )
        embed_dim = probe_embed_dim

    classifier = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        num_classes=num_classes,
    )

    msg = classifier.load_state_dict(state, strict=True)
    logger.info(f"Classifier loaded (embed_dim={embed_dim}): {msg}")

    classifier.to(device).eval()
    for p in classifier.parameters():
        p.requires_grad = False
    del ckpt
    return classifier


# ──────────────────────────────────────────────────────────────────────
#  Clip extraction
# ──────────────────────────────────────────────────────────────────────
def extract_clips(video_path, frames_per_clip, frame_step):
    """
    Slide a non-overlapping temporal window across the video.
    Returns list of dicts: {frames: np.ndarray[T,H,W,C],
                            start_sec, end_sec, fps}.
    """
    try:
        vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
    except Exception as e:
        logger.error(f"Cannot open {video_path}: {e}")
        return []

    n_frames = len(vr)
    try:
        fps = math.ceil(vr.get_avg_fps())
    except Exception:
        fps = 30

    clip_span = frames_per_clip * frame_step  # frames covered by 1 clip

    if n_frames < frames_per_clip:
        logger.warning(f"Video too short ({n_frames} frames): {video_path}")
        return []

    clips = []
    start = 0
    while start + clip_span <= n_frames:
        indices = np.linspace(
            start, start + clip_span - 1, num=frames_per_clip
        ).astype(np.int64)
        indices = np.clip(indices, 0, n_frames - 1)
        frames = vr.get_batch(indices.tolist()).asnumpy()  # [T,H,W,C]
        clips.append(
            {
                "frames": frames,
                "start_sec": start / fps,
                "end_sec": (start + clip_span) / fps,
                "fps": fps,
            }
        )
        start += clip_span

    # Handle tail: if remaining frames ≥ frames_per_clip, grab a last clip
    if start < n_frames and (n_frames - start) >= frames_per_clip:
        tail_span = min(clip_span, n_frames - start)
        indices = np.linspace(
            start, start + tail_span - 1, num=frames_per_clip
        ).astype(np.int64)
        indices = np.clip(indices, 0, n_frames - 1)
        frames = vr.get_batch(indices.tolist()).asnumpy()
        clips.append(
            {
                "frames": frames,
                "start_sec": start / fps,
                "end_sec": (start + tail_span) / fps,
                "fps": fps,
            }
        )

    return clips


# ──────────────────────────────────────────────────────────────────────
#  Inference helpers
# ──────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def run_batch(encoder, classifier, batch_tensor, device):
    """
    Run encoder + classifier on a batch of clips.
    batch_tensor: [B, C, T, H, W]
    Returns: predictions [B], probabilities [B, num_classes]
    """
    x = batch_tensor.to(device)
    features = encoder(x)  # [B, N, D]
    logits = classifier(features)  # [B, num_classes]
    probs = F.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    return preds.cpu(), probs.cpu()


# ──────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Compute TP, TN, FP, FN, accuracy, precision, recall, F1."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "total_clips": n,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    w = 50
    print("\n" + "=" * w)
    print("  EVALUATION RESULTS")
    print("=" * w)
    print(f"  Total clips evaluated : {metrics['total_clips']}")
    print("-" * w)
    print(f"  True  Positives (TP)  : {metrics['TP']}")
    print(f"  True  Negatives (TN)  : {metrics['TN']}")
    print(f"  False Positives (FP)  : {metrics['FP']}")
    print(f"  False Negatives (FN)  : {metrics['FN']}")
    print("-" * w)
    print(f"  Accuracy              : {metrics['accuracy']:.4f}")
    print(f"  Precision             : {metrics['precision']:.4f}")
    print(f"  Recall                : {metrics['recall']:.4f}")
    print(f"  F1 Score              : {metrics['f1']:.4f}")
    print("-" * w)
    cm = metrics["confusion_matrix"]
    print("  Confusion Matrix:")
    print("                     Pred Non-Violent  Pred Violent")
    print(f"    True Non-Violent       {cm[0][0]:>6}         {cm[0][1]:>6}")
    print(f"    True Violent           {cm[1][0]:>6}         {cm[1][1]:>6}")
    print("=" * w + "\n")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover dataset ──
    samples = discover_dataset(args.dataset_dir)
    if args.max_videos is not None:
        samples = samples[: args.max_videos]
    logger.info(
        f"Found {len(samples)} videos "
        f"(violent: {sum(1 for s in samples if s['class_label'] == 1)}, "
        f"non-violent: {sum(1 for s in samples if s['class_label'] == 0)})"
    )

    # ── Build transform ──
    transform = build_eval_transform(args.resolution)

    # ── Load models ──
    # Peek at probe to detect its embed_dim — this determines which encoder
    # architecture to use (the probe was trained with a specific encoder).
    probe_embed_dim = _detect_probe_embed_dim(args.probe_ckpt)
    if probe_embed_dim:
        logger.info(f"Probe embed_dim={probe_embed_dim}")

    encoder = load_encoder(
        ckpt_path=args.encoder_ckpt,
        model_name=args.model_name,
        checkpoint_key=args.checkpoint_key,
        resolution=args.resolution,
        frames_per_clip=args.frames_per_clip,
        device=device,
        probe_embed_dim=probe_embed_dim,
    )
    classifier = load_classifier(
        ckpt_path=args.probe_ckpt,
        embed_dim=encoder.embed_dim,
        num_heads=args.num_heads,
        depth=args.probe_depth,
        num_classes=args.num_classes,
        device=device,
    )
    logger.info(
        f"Encoder embed_dim={encoder.embed_dim}, "
        f"Classifier classes={args.num_classes}"
    )

    # ── Inference loop ──
    all_gt, all_pred, all_prob = [], [], []
    clip_results = []  # detailed per-clip results

    t0 = time.time()
    for vi, sample in enumerate(samples):
        video_path = sample["video_path"]
        label_path = sample["label_path"]
        class_label = sample["class_label"]
        video_name = sample["video_name"]

        # Parse temporal annotations
        intervals = parse_temporal_labels(label_path)

        # Extract clips
        clips = extract_clips(
            video_path, args.frames_per_clip, args.frame_step
        )
        if not clips:
            logger.warning(f"[{vi + 1}/{len(samples)}] No clips: {video_name}")
            continue

        # Build ground-truth labels for each clip
        clip_labels = []
        for c in clips:
            if class_label == 0:
                # Non-violent video → all clips are non-violent
                clip_labels.append(0)
            else:
                # Violent video → check temporal overlap
                if intervals:
                    is_v = clip_is_violent(
                        c["start_sec"],
                        c["end_sec"],
                        intervals,
                        args.overlap_threshold,
                    )
                    clip_labels.append(1 if is_v else 0)
                else:
                    # Violent class but no temporal labels → label entire
                    # video as violent
                    clip_labels.append(1)

        # Apply transform to each clip's frames
        clip_tensors = []
        for c in clips:
            t = transform(c["frames"])  # [C, T, H, W]
            clip_tensors.append(t)

        # Batch inference
        preds_list, probs_list = [], []
        for bi in range(0, len(clip_tensors), args.batch_size):
            batch = torch.stack(
                clip_tensors[bi : bi + args.batch_size], dim=0
            )  # [B, C, T, H, W]
            preds, probs = run_batch(encoder, classifier, batch, device)
            preds_list.append(preds)
            probs_list.append(probs)

        preds_all = torch.cat(preds_list).numpy()
        probs_all = torch.cat(probs_list).numpy()

        # Collect results
        for ci, (c, gt, pred, prob) in enumerate(
            zip(clips, clip_labels, preds_all, probs_all)
        ):
            all_gt.append(gt)
            all_pred.append(int(pred))
            all_prob.append(prob.tolist())
            clip_results.append(
                {
                    "video": video_name,
                    "class": sample["class_name"],
                    "clip_idx": ci,
                    "start_sec": f"{c['start_sec']:.2f}",
                    "end_sec": f"{c['end_sec']:.2f}",
                    "gt_label": gt,
                    "pred_label": int(pred),
                    "prob_nonviolent": f"{prob[0]:.4f}",
                    "prob_violent": f"{prob[1]:.4f}",
                    "correct": int(gt == int(pred)),
                }
            )

        elapsed = time.time() - t0
        n_clips_so_far = len(all_gt)
        logger.info(
            f"[{vi + 1}/{len(samples)}] {video_name}: "
            f"{len(clips)} clips, "
            f"total={n_clips_so_far}, "
            f"elapsed={elapsed:.1f}s"
        )

    # ── Compute metrics ──
    if not all_gt:
        logger.error("No clips were processed!")
        return

    metrics = compute_metrics(all_gt, all_pred)
    print_metrics(metrics)

    # ── Save detailed CSV ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"clip_results_{timestamp}.csv")
    fieldnames = list(clip_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clip_results)
    logger.info(f"Detailed clip results saved to {csv_path}")

    # ── Save summary metrics ──
    summary_path = os.path.join(args.output_dir, f"metrics_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write("V-JEPA 2.1 Weapon Detection Evaluation\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Dataset: {args.dataset_dir}\n")
        f.write(f"Encoder: {args.encoder_ckpt}\n")
        f.write(f"Probe:   {args.probe_ckpt}\n")
        f.write(f"Resolution: {args.resolution}\n")
        f.write(
            f"Frames/clip: {args.frames_per_clip}, "
            f"Frame step: {args.frame_step}\n\n"
        )
        for k in [
            "total_clips",
            "TP",
            "TN",
            "FP",
            "FN",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]:
            v = metrics[k]
            f.write(
                f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n"
            )
        cm = metrics["confusion_matrix"]
        f.write("\nConfusion Matrix:\n")
        f.write(f"  [[TN={cm[0][0]}, FP={cm[0][1]}],\n")
        f.write(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]\n")
    logger.info(f"Summary metrics saved to {summary_path}")

    total_time = time.time() - t0
    logger.info(
        f"Total inference time: {total_time:.1f}s "
        f"({len(all_gt)} clips, "
        f"{len(all_gt) / total_time:.1f} clips/s)"
    )


if __name__ == "__main__":
    main()
