"""Combined YOLOv26 + V-JEPA 2.1 Evaluation Pipeline.

Evaluates weapon (gun) detection by combining YOLOv26 (per-frame object
detection) with V-JEPA 2.1 (16-frame violence classification).  V-JEPA acts
as a supplementary signal to reduce false positives on ambiguous / weak
YOLO detections — it does NOT override strong YOLO detections.

Usage:
    python combination-eval.py \
        --dataset ./dataset \
        --encoder_weights pretrained-models/vjepa2_1_vitl_dist_vitG_384.pt \
        --probe_weights trained-probes/vitl-probe.pt \
        --yolo26-checkpoint trained-models/yolo26m-weapon-det.pt \
        --yolo-threshold 0.3 \
        --vjepa-threshold 0.5 \
        --strong-ratio 0.5 \
        --output results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import glob
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

import src.datasets.utils.video.transforms as video_transforms  # type: ignore
import src.datasets.utils.video.volume_transforms as volume_transforms  # type: ignore

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_labels(csv_path: str) -> List[Tuple[float, float]]:
    segments = []
    if not os.path.isfile(csv_path):
        return segments

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                start = float(row[0].strip())
                end = float(row[1].strip())
                segments.append((start, end))
            except ValueError:
                continue
    return segments


def clip_overlaps_any_label(
    clip_start_sec: float,
    clip_end_sec: float,
    labels: List[Tuple[float, float]],
) -> bool:
    """Return True if the clip's time range overlaps any labeled segment."""
    for label_start, label_end in labels:
        if clip_start_sec <= label_end and clip_end_sec >= label_start:
            return True
    return False


def evaluate_clip_yolo(
    frames: List[np.ndarray],
    yolo_model: YOLO,
    yolo_threshold: float,
) -> Dict[str, Any]:
    """Run YOLO on every frame in the clip.

    Returns a dict with:
        yolo_frame_count  - number of frames with ≥1 gun detection
        yolo_frame_ratio  - frame_count / total_frames
        yolo_max_conf     - max confidence across all frames
        yolo_avg_conf     - mean confidence across frames WITH detections
    """
    total = len(frames)
    frame_count = 0
    max_conf = 0.0
    conf_sum = 0.0

    for frame in frames:
        results = yolo_model.predict(frame, verbose=False, conf=yolo_threshold)
        frame_has_det = False
        frame_max = 0.0
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                frame_has_det = True
                c = float(r.boxes.conf.max())
                frame_max = max(frame_max, c)

        if frame_has_det:
            frame_count += 1
            max_conf = max(max_conf, frame_max)
            conf_sum += frame_max

    avg_conf = conf_sum / frame_count if frame_count > 0 else 0.0

    return {
        "yolo_frame_count": frame_count,
        "yolo_frame_ratio": frame_count / total if total > 0 else 0.0,
        "yolo_max_conf": max_conf,
        "yolo_avg_conf": avg_conf,
    }


def evaluate_clip_vjepa(
    frames,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    model_positive_idx: int,
    device: str,
) -> Dict[str, Any]:
    """Run V-JEPA 2.1 on a 16-frame clip.

    Args:
        frames: list of HWC uint8 numpy arrays (RGB), or an ndarray / tensor
            of shape [T, H, W, C].  Matches what inference.py's transforms
            expect — see inference.py:130-187.
        model_positive_idx: index of the positive (violent) class in the
            classifier output.

    Returns a dict with:
        vjepa_label        - top-1 predicted class index (int)
        vjepa_conf         - softmax probability of the top-1 class (0-1)
        vjepa_violent_conf - softmax probability of the positive class (0-1)
    """
    with torch.inference_mode():
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()

        frame_list = [frames[i] for i in range(len(frames))]

        img_size = 384
        if hasattr(encoder, "patch_embed") and hasattr(
            encoder.patch_embed, "img_size"
        ):
            if isinstance(encoder.patch_embed.img_size, tuple):
                img_size = encoder.patch_embed.img_size[0]
            else:
                img_size = encoder.patch_embed.img_size

        short_side_size = int(256.0 / 224 * img_size)
        eval_transform = video_transforms.Compose(
            [
                video_transforms.Resize(
                    short_side_size, interpolation="bilinear"
                ),
                video_transforms.CenterCrop(size=(img_size, img_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        x_pt = eval_transform(frame_list).to(device).unsqueeze(0)
        out_patch_features_pt = encoder(x_pt)
        out_classifier = classifier(out_patch_features_pt)

        probs = F.softmax(out_classifier[0], dim=0)
        label_idx = int(out_classifier.argmax(dim=1).item())

    return {
        "vjepa_label": label_idx,
        "vjepa_conf": float(probs[label_idx].item()),
        "vjepa_violent_conf": float(probs[model_positive_idx].item()),
    }


def strategy_yolo_only(
    clip_result: Dict[str, Any],
) -> Tuple[bool, float]:
    """Baseline: any YOLO detection in the clip → POSITIVE."""
    detected = clip_result["yolo_frame_count"] > 0
    conf = clip_result["yolo_max_conf"] if detected else 0.0
    return detected, conf


def strategy_vjepa_only(
    clip_result: Dict[str, Any],
    model_positive_idx: int,
) -> Tuple[bool, float]:
    """Baseline: V-JEPA top-1 == positive class → POSITIVE."""
    is_violent = clip_result["vjepa_label"] == model_positive_idx
    return is_violent, clip_result["vjepa_violent_conf"]


def strategy_frame_gated(
    clip_result: Dict[str, Any],
    strong_ratio: float = 0.5,
    vjepa_threshold: float = 0.5,
) -> Tuple[bool, float]:
    """Frame-count gated.

    - Strong YOLO (ratio >= strong_ratio) → always POSITIVE
    - Weak YOLO (0 < ratio < strong_ratio) → V-JEPA decides
    - No YOLO → NEGATIVE
    """
    ratio = clip_result["yolo_frame_ratio"]
    if ratio >= strong_ratio:
        return True, clip_result["yolo_max_conf"]
    elif ratio > 0:
        if clip_result["vjepa_violent_conf"] >= vjepa_threshold:
            return True, clip_result["yolo_max_conf"]
        else:
            return False, 0.0
    else:
        return False, 0.0


def strategy_conf_gated(
    clip_result: Dict[str, Any],
    high_conf: float = 0.8,
    low_conf: float = 0.3,
    vjepa_threshold: float = 0.5,
) -> Tuple[bool, float]:
    """Confidence-gated.

    - High-confidence YOLO (>= high_conf) --> always POSITIVE
    - Mid-confidence YOLO (>= low_conf)   --> V-JEPA decides
    - Below low_conf                      --> NEGATIVE
    """
    yolo_conf = clip_result["yolo_max_conf"]

    if yolo_conf >= high_conf:
        return True, yolo_conf
    elif yolo_conf >= low_conf:
        if clip_result["vjepa_violent_conf"] >= vjepa_threshold:
            return True, yolo_conf
        else:
            return False, 0.0
    else:
        return False, 0.0


def strategy_weighted(
    clip_result: Dict[str, Any],
    w_yolo: float = 0.7,
    w_vjepa: float = 0.3,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """Weighted score combining YOLO strength and V-JEPA context.

    score = w_yolo * (yolo_max_conf * yolo_frame_ratio) + w_vjepa * vjepa_violent_conf

    When YOLO is strong, the first term dominates and V-JEPA barely
    matters.  When YOLO is weak, V-JEPA becomes the deciding factor.
    """
    yolo_score = clip_result["yolo_max_conf"] * clip_result["yolo_frame_ratio"]
    vjepa_score = clip_result["vjepa_violent_conf"]

    combined = w_yolo * yolo_score + w_vjepa * vjepa_score
    return combined >= threshold, combined


def compute_metrics(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> Dict[str, float]:
    """Compute evaluation metrics from a confusion matrix."""
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if (4 * precision + recall) > 0
        else 0.0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "specificity": round(specificity, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "accuracy": round(accuracy, 4),
    }


def find_videos(directory: str) -> List[str]:
    """Find all .mp4 and .avi files in *directory* (non-recursive)."""
    videos: List[str] = []
    for ext in ("*.mp4", "*.avi"):
        videos.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(videos)


def get_label_path(video_path: str, labels_dir: str) -> str:
    """Return the matching label CSV path for a video file."""
    stem = Path(video_path).stem
    return os.path.join(labels_dir, f"{stem}.csv")


STRATEGY_NAMES = [
    "yolo_only",
    "vjepa_only",
    "frame_gated",
    "conf_gated",
    "weighted",
]


def evaluate_dataset(
    dataset_dir: str,
    yolo_model: YOLO,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    model_positive_idx: int,
    class_label_names: List[str],
    device: str,
    *,
    yolo_threshold: float,
    vjepa_threshold: float,
    strong_ratio: float,
    high_conf: float,
    low_conf: float,
    w_yolo: float,
    w_vjepa: float,
    weighted_threshold: float,
    num_frames: int,
    frame_step: int,
) -> Dict[str, Any]:
    """Walk the dataset, evaluate every clip, and return the full report."""

    # Confusion-matrix counters per strategy
    counters = {
        s: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for s in STRATEGY_NAMES
    }

    per_video_results = []
    total_clips = 0
    positive_clips = 0
    negative_clips = 0

    # Iterate over both folders
    for folder_name, is_positive_folder in [
        ("violent", True),
        ("non-violent", False),
    ]:
        videos_dir = os.path.join(dataset_dir, folder_name, "videos")
        labels_dir = os.path.join(dataset_dir, folder_name, "labels")

        if not os.path.isdir(videos_dir):
            logger.warning(f"Videos directory not found: {videos_dir}")
            continue

        video_paths = find_videos(videos_dir)
        logger.info(
            f"Found {len(video_paths)} videos in {folder_name}/videos/"
        )

        for vi, video_path in enumerate(video_paths):
            video_name = os.path.relpath(video_path, dataset_dir)
            logger.info(f"  [{vi + 1}/{len(video_paths)}] {video_name}")

            # Load temporal annotations
            label_path = get_label_path(video_path, labels_dir)
            labels = load_labels(label_path)

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Cannot open: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            video_result: Dict[str, Any] = {
                "video": video_name,
                "folder": folder_name,
                "fps": fps,
                "total_frames": total_frames,
                "label_segments": [{"start": s, "end": e} for s, e in labels],
                "clips": [],
            }

            # Extract non-overlapping strided clips (matches inference.py:
            # num_frames sampled every frame_step-th frame from a
            # num_frames * frame_step raw-frame window).
            raw_frames_per_clip = num_frames * frame_step
            clip_idx = 0
            for clip_start_frame in range(
                0, total_frames, raw_frames_per_clip
            ):
                clip_end_frame = clip_start_frame + raw_frames_per_clip - 1
                if clip_end_frame >= total_frames:
                    break

                clip_start_sec = clip_start_frame / fps
                clip_end_sec = clip_end_frame / fps

                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)
                sampled_bgr_frames: List[np.ndarray] = []
                sampled_rgb_frames: List[np.ndarray] = []
                for offset in range(raw_frames_per_clip):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if offset % frame_step == 0:
                        sampled_bgr_frames.append(frame)
                        sampled_rgb_frames.append(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        )

                if len(sampled_bgr_frames) < num_frames:
                    break  # tail short of a full clip

                # Ground truth
                if is_positive_folder and clip_overlaps_any_label(
                    clip_start_sec, clip_end_sec, labels
                ):
                    ground_truth = 1
                    positive_clips += 1
                else:
                    ground_truth = 0
                    negative_clips += 1
                total_clips += 1

                # YOLO inference (per-frame, on the strided frames)
                yolo_result = evaluate_clip_yolo(
                    sampled_bgr_frames, yolo_model, yolo_threshold
                )

                # V-JEPA 2.1 inference (whole clip, same strided frames)
                vjepa_result = evaluate_clip_vjepa(
                    sampled_rgb_frames,
                    encoder,
                    classifier,
                    model_positive_idx,
                    device,
                )

                clip_result = {**yolo_result, **vjepa_result}

                # Apply all strategies
                predictions: Dict[str, Tuple[bool, float]] = {
                    "yolo_only": strategy_yolo_only(clip_result),
                    "vjepa_only": strategy_vjepa_only(
                        clip_result, model_positive_idx
                    ),
                    "frame_gated": strategy_frame_gated(
                        clip_result, strong_ratio, vjepa_threshold
                    ),
                    "conf_gated": strategy_conf_gated(
                        clip_result, high_conf, low_conf, vjepa_threshold
                    ),
                    "weighted": strategy_weighted(
                        clip_result, w_yolo, w_vjepa, weighted_threshold
                    ),
                }

                # Update confusion matrices
                clip_outcomes = {}
                for sname in STRATEGY_NAMES:
                    predicted, confidence = predictions[sname]
                    pred_label = 1 if predicted else 0

                    if ground_truth == 1 and pred_label == 1:
                        outcome = "TP"
                    elif ground_truth == 0 and pred_label == 1:
                        outcome = "FP"
                    elif ground_truth == 0 and pred_label == 0:
                        outcome = "TN"
                    else:
                        outcome = "FN"

                    counters[sname][outcome] += 1
                    clip_outcomes[sname] = {
                        "predicted": pred_label,
                        "confidence": round(confidence, 4),
                        "outcome": outcome,
                    }

                # Log clip detail
                video_result["clips"].append(
                    {
                        "clip_idx": clip_idx,
                        "start_frame": clip_start_frame,
                        "end_frame": clip_end_frame,
                        "start_sec": round(clip_start_sec, 2),
                        "end_sec": round(clip_end_sec, 2),
                        "ground_truth": ground_truth,
                        "yolo_frame_count": yolo_result["yolo_frame_count"],
                        "yolo_frame_ratio": round(
                            yolo_result["yolo_frame_ratio"], 4
                        ),
                        "yolo_max_conf": round(
                            yolo_result["yolo_max_conf"], 4
                        ),
                        "yolo_avg_conf": round(
                            yolo_result["yolo_avg_conf"], 4
                        ),
                        "vjepa_label": vjepa_result["vjepa_label"],
                        "vjepa_label_name": class_label_names[
                            vjepa_result["vjepa_label"]
                        ],
                        "vjepa_violent_conf": round(
                            vjepa_result["vjepa_violent_conf"], 4
                        ),
                        "strategies": clip_outcomes,
                    }
                )
                clip_idx += 1

            cap.release()
            per_video_results.append(video_result)

    # Compute metrics per strategy
    results: Dict[str, Any] = {}
    for sname in STRATEGY_NAMES:
        cm = counters[sname]
        metrics = compute_metrics(cm["TP"], cm["FP"], cm["TN"], cm["FN"])
        results[sname] = {"confusion_matrix": cm, "metrics": metrics}

    # FP reduction analysis
    yolo_fp = counters["yolo_only"]["FP"]
    combined_strategies = ["frame_gated", "conf_gated", "weighted"]
    best_strategy = min(combined_strategies, key=lambda s: counters[s]["FP"])
    best_combined_fp = counters[best_strategy]["FP"]

    fp_reduction_pct = (
        round((yolo_fp - best_combined_fp) / yolo_fp * 100, 1)
        if yolo_fp > 0
        else 0.0
    )

    return {
        "dataset_stats": {
            "total_videos": len(per_video_results),
            "violent_videos": sum(
                1 for v in per_video_results if v["folder"] == "violent"
            ),
            "non_violent_videos": sum(
                1 for v in per_video_results if v["folder"] == "non-violent"
            ),
            "total_clips": total_clips,
            "positive_clips": positive_clips,
            "negative_clips": negative_clips,
        },
        "results": results,
        "fp_reduction": {
            "yolo_only_fp": yolo_fp,
            "best_combined_fp": best_combined_fp,
            "best_strategy": best_strategy,
            "fp_saved_by_vjepa": yolo_fp - best_combined_fp,
            "fp_reduction_pct": fp_reduction_pct,
        },
        "per_video": per_video_results,
    }


def print_summary(report: Dict[str, Any]) -> None:
    """Print a formatted comparison table to stdout."""
    stats = report["dataset_stats"]

    print("\n" + "=" * 78)
    print("  COMBINED YOLOv26 + V-JEPA EVALUATION RESULTS")
    print("=" * 78)
    print(
        f"  Videos : {stats['total_videos']:>5}  "
        f"({stats['violent_videos']} violent, "
        f"{stats['non_violent_videos']} non-violent)"
    )
    print(
        f"  Clips  : {stats['total_clips']:>5}  "
        f"({stats['positive_clips']} positive, "
        f"{stats['negative_clips']} negative)"
    )
    cls_str = ", ".join(report["config"]["class_labels"])
    print(
        f"  V-JEPA classes : {cls_str}  "
        f"(positive_idx={report['config']['positive_idx']})"
    )
    print("-" * 78)

    # Header
    print(
        f"  {'Strategy':<16}"
        f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'F2':>6} {'Spec':>6}"
    )
    print("  " + "-" * 74)

    for name in STRATEGY_NAMES:
        r = report["results"][name]
        cm = r["confusion_matrix"]
        m = r["metrics"]
        print(
            f"  {name:<16}"
            f"{cm['TP']:>5} {cm['FP']:>5} {cm['TN']:>5} {cm['FN']:>5}  "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['f2']:>6.3f} {m['specificity']:>6.3f}"
        )

    # FP reduction
    fp = report["fp_reduction"]
    print("-" * 78)
    print(
        f"  FP Reduction : YOLO-only FP = {fp['yolo_only_fp']} → "
        f"Best combined FP = {fp['best_combined_fp']} "
        f"({fp['best_strategy']}, "
        f"-{fp['fp_reduction_pct']}%)"
    )
    print(f"  V-JEPA saved {fp['fp_saved_by_vjepa']} false positive(s)")
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combined YOLOv26 + V-JEPA evaluation pipeline. "
            "Compares weapon-detection strategies and reports "
            "TP / FP / TN / FN with full metrics."
        ),
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="/tf/data/test-dataset",
        help=(
            "Path to eval dataset root containing violent/ and "
            "non-violent/ subdirectories"
        ),
    )

    # V-JEPA 2.1 model paths
    parser.add_argument(
        "--encoder_weights",
        type=str,
        default="/tf/data/pretrained-models/vjepa2_1_vitG_384.pt",
        help="Path to V-JEPA 2.1 encoder weights",
    )
    parser.add_argument(
        "--probe_weights",
        type=str,
        default="data/outputs/evals_2_1/vitG-384/weaponized_2cls/video_classification_frozen/weaponized-2cls-vitg16-384/best.pt",
        help="Path to attentive probe weights",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames per clip fed to V-JEPA",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=4,
        help="Frame sampling stride (must match training config frame_step)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="V-JEPA encoder input image size",
    )

    # YOLO model path
    parser.add_argument(
        "--yolo26-checkpoint",
        type=str,
        default="/tf/data/pretrained-models/yolo26m-weapon-det.pt",
        help="Path to YOLOv26 gun detection checkpoint",
    )

    # Thresholds / strategy parameters
    parser.add_argument(
        "--yolo-threshold",
        type=float,
        default=0.41,
        help="YOLO confidence threshold for per-frame detection",
    )
    parser.add_argument(
        "--vjepa-threshold",
        type=float,
        default=0.5,
        help="V-JEPA violent-class confidence threshold (tiebreaker)",
    )
    parser.add_argument(
        "--strong-ratio",
        type=float,
        default=0.5,
        help=(
            "Frame-ratio threshold above which YOLO is considered "
            "'strong' and V-JEPA is ignored (frame_gated strategy)"
        ),
    )
    parser.add_argument(
        "--high-conf",
        type=float,
        default=0.8,
        help="High-confidence YOLO threshold (conf_gated strategy)",
    )
    parser.add_argument(
        "--low-conf",
        type=float,
        default=0.3,
        help="Low-confidence YOLO threshold (conf_gated strategy)",
    )
    parser.add_argument(
        "--w-yolo",
        type=float,
        default=0.7,
        help="YOLO weight for weighted strategy",
    )
    parser.add_argument(
        "--w-vjepa",
        type=float,
        default=0.3,
        help="V-JEPA weight for weighted strategy",
    )
    parser.add_argument(
        "--weighted-threshold",
        type=float,
        default=0.5,
        help="Score threshold for weighted strategy",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Path to output JSON report",
    )
    args = parser.parse_args()

    # Validate dataset directory
    for sub in ("violent/videos", "non-violent/videos"):
        p = os.path.join(args.dataset, sub)
        if not os.path.isdir(p):
            raise SystemExit(f"Error: expected directory not found: {p}")

    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load V-JEPA 2.1 encoder (arch auto-detected from filename)
    logger.info("Loading V-JEPA 2.1 encoder …")
    weight_basename = os.path.basename(args.encoder_weights).lower()
    weight_parts = weight_basename.split("_")
    arch_part = weight_parts[2] if len(weight_parts) > 2 else ""

    if "vitg" in arch_part:
        from src.models.vision_transformer import (  # type: ignore
            vit_gigantic_xformers as vit_model,
        )
    elif "vitl" in arch_part:
        from src.models.vision_transformer import vit_large as vit_model  # type: ignore
    else:
        from src.models.vision_transformer import vit_base as vit_model  # type: ignore

    encoder = vit_model(
        img_size=args.img_size,
        num_frames=args.num_frames,
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )
    pretrained_dict = torch.load(
        args.encoder_weights, map_location="cpu", weights_only=True
    )
    if "ema_encoder" in pretrained_dict:
        pretrained_dict = pretrained_dict["ema_encoder"]
    elif "target_encoder" in pretrained_dict:
        pretrained_dict = pretrained_dict["target_encoder"]
    elif "encoder" in pretrained_dict:
        pretrained_dict = pretrained_dict["encoder"]

    pretrained_dict = {
        k.replace("module.", ""): v for k, v in pretrained_dict.items()
    }
    pretrained_dict = {
        k.replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }
    encoder.load_state_dict(pretrained_dict, strict=False)
    encoder.to(device).eval()

    # Load attentive classifier probe
    logger.info("Loading attentive classifier probe …")
    from src.models.attentive_pooler import AttentiveClassifier  # type: ignore

    probe_dict = torch.load(
        args.probe_weights, map_location="cpu", weights_only=True
    )
    if "classifiers" in probe_dict:
        probe_dict = probe_dict["classifiers"][0]
    probe_dict = {k.replace("module.", ""): v for k, v in probe_dict.items()}

    model_num_classes = probe_dict["linear.weight"].shape[0]
    if model_num_classes == 2:
        model_positive_idx = 1
        class_label_names = ["non-violent", "violent"]
        logger.info("Detected 2-class probe: [non-violent, violent]")
    elif model_num_classes == 3:
        model_positive_idx = 2
        class_label_names = ["non-violent", "fighting", "violent"]
        logger.info(
            "Detected 3-class probe: [non-violent, fighting, violent] "
            "(fighting counted as negative)"
        )
    else:
        model_positive_idx = model_num_classes - 1
        class_label_names = [
            f"class_{i}" for i in range(model_num_classes)
        ]
        logger.warning(
            f"Detected {model_num_classes}-class probe; "
            f"using positive_idx={model_positive_idx}"
        )

    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=16,
        depth=4,
        num_classes=model_num_classes,
    )
    msg = classifier.load_state_dict(probe_dict, strict=False)
    if msg.missing_keys:
        logger.warning(
            f"Missing keys in probe checkpoint: {msg.missing_keys}"
        )
    if msg.unexpected_keys:
        logger.warning(
            f"Unexpected keys in probe checkpoint: {msg.unexpected_keys}"
        )
    if not msg.missing_keys and not msg.unexpected_keys:
        logger.info("Probe checkpoint loaded successfully (all keys matched).")
    classifier.to(device).eval()

    # Load YOLOv26
    logger.info("Loading YOLOv26 gun detector …")
    yolo_model = YOLO(args.yolo26_checkpoint)

    # Run evaluation
    logger.info("Starting evaluation …")
    t_start = time.time()

    report = evaluate_dataset(
        dataset_dir=args.dataset,
        yolo_model=yolo_model,
        encoder=encoder,
        classifier=classifier,
        model_positive_idx=model_positive_idx,
        class_label_names=class_label_names,
        device=device,
        yolo_threshold=args.yolo_threshold,
        vjepa_threshold=args.vjepa_threshold,
        strong_ratio=args.strong_ratio,
        high_conf=args.high_conf,
        low_conf=args.low_conf,
        w_yolo=args.w_yolo,
        w_vjepa=args.w_vjepa,
        weighted_threshold=args.weighted_threshold,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
    )
    elapsed = time.time() - t_start

    # Attach config to report
    report["config"] = {
        "dataset": os.path.abspath(args.dataset),
        "yolo_checkpoint": args.yolo26_checkpoint,
        "encoder_weights": args.encoder_weights,
        "probe_weights": args.probe_weights,
        "yolo_threshold": args.yolo_threshold,
        "vjepa_threshold": args.vjepa_threshold,
        "strong_ratio": args.strong_ratio,
        "high_conf": args.high_conf,
        "low_conf": args.low_conf,
        "w_yolo": args.w_yolo,
        "w_vjepa": args.w_vjepa,
        "weighted_threshold": args.weighted_threshold,
        "num_frames": args.num_frames,
        "frame_step": args.frame_step,
        "img_size": args.img_size,
        "raw_frames_per_clip": args.num_frames * args.frame_step,
        "num_classes": model_num_classes,
        "positive_idx": model_positive_idx,
        "class_labels": class_label_names,
        "device": device,
        "elapsed_sec": round(elapsed, 1),
    }

    # Print summary
    print_summary(report)

    # Sanity check
    for sname in STRATEGY_NAMES:
        cm = report["results"][sname]["confusion_matrix"]
        total = cm["TP"] + cm["FP"] + cm["TN"] + cm["FN"]
        assert total == report["dataset_stats"]["total_clips"], (
            f"Sanity check failed for {sname}: "
            f"TP+FP+TN+FN={total} != total_clips="
            f"{report['dataset_stats']['total_clips']}"
        )

    # Write JSON report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Full report saved to: {args.output}")
    logger.info(f"Evaluation completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
