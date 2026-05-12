"""V-JEPA vs YOLO Prediction Overlap Analysis.

Runs both models on every clip and analyzes where their predictions
agree/disagree to recommend: use YOLO only, V-JEPA only, or stack both.

Usage:
    python pred-overlapping.py --dataset ./dataset \
        --encoder_weights pretrained-models/vjepa2_1_vitl_dist_vitG_384.pt \
        --probe_weights trained-probes/vitl-probe.pt \
        --yolo26-checkpoint trained-models/yolo26m-weapon-det.pt
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
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                segments.append((float(row[0].strip()), float(row[1].strip())))
            except ValueError:
                continue
    return segments


def clip_overlaps_any_label(
    start: float, end: float, labels: List[Tuple[float, float]]
) -> bool:
    for ls, le in labels:
        if start <= le and end >= ls:
            return True
    return False


def find_videos(directory: str) -> List[str]:
    videos: List[str] = []
    for ext in ("*.mp4", "*.avi"):
        videos.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(videos)


def evaluate_clip_yolo(
    frames: List[np.ndarray],
    yolo_model: YOLO,
    threshold: float,
) -> Dict[str, Any]:
    """Full-frame weapon detection on every frame."""
    total = len(frames)
    frame_count = 0
    max_conf = 0.0
    conf_sum = 0.0
    for frame in frames:
        results = yolo_model.predict(frame, verbose=False, conf=threshold)
        frame_max = 0.0
        hit = False
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                hit = True
                frame_max = max(frame_max, float(r.boxes.conf.max()))
        if hit:
            frame_count += 1
            max_conf = max(max_conf, frame_max)
            conf_sum += frame_max
    return {
        "yolo_frame_count": frame_count,
        "yolo_frame_ratio": frame_count / total if total else 0.0,
        "yolo_max_conf": max_conf,
        "yolo_avg_conf": conf_sum / frame_count if frame_count else 0.0,
    }


def evaluate_clip_vjepa(
    frames,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    positive_idx: int,
    device: str,
) -> Dict[str, Any]:
    """V-JEPA 2.1 clip-level classification."""
    with torch.inference_mode():
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        frame_list = [frames[i] for i in range(len(frames))]
        img_size = 384
        if hasattr(encoder, "patch_embed") and hasattr(
            encoder.patch_embed, "img_size"
        ):
            s = encoder.patch_embed.img_size
            img_size = s[0] if isinstance(s, tuple) else s
        short_side = int(256.0 / 224 * img_size)
        tfm = video_transforms.Compose(
            [
                video_transforms.Resize(short_side, interpolation="bilinear"),
                video_transforms.CenterCrop(size=(img_size, img_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        x = tfm(frame_list).to(device).unsqueeze(0)
        logits = classifier(encoder(x))
        probs = F.softmax(logits[0], dim=0)
        label = int(logits.argmax(dim=1).item())
    return {
        "vjepa_label": label,
        "vjepa_conf": float(probs[label].item()),
        "vjepa_violent_conf": float(probs[positive_idx].item()),
    }


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    total = tp + fp + tn + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "specificity": round(spec, 4),
        "fpr": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "fnr": round(fn / (fn + tp), 4) if (fn + tp) else 0.0,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
    }


def cohens_kappa(table: Dict[str, int], n: int) -> float:
    """Cohen's kappa from a 2x2 agreement table."""
    if n == 0:
        return 0.0
    a, b, c, d = (
        table["both_neg"],
        table["yolo_only"],
        table["vjepa_only"],
        table["both_pos"],
    )
    po = (a + d) / n
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    pe = (row1 * col1 + row2 * col2) / (n * n)
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0


def collect_predictions(
    dataset_dir: str,
    yolo_model: YOLO,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    positive_idx: int,
    device: str,
    *,
    yolo_threshold: float,
    num_frames: int,
    frame_step: int,
) -> List[Dict[str, Any]]:
    """Run both models on every clip. Return list of per-clip records."""
    clips: List[Dict[str, Any]] = []
    raw_per_clip = num_frames * frame_step

    for folder, is_pos_folder in [("violent", True), ("non-violent", False)]:
        vdir = os.path.join(dataset_dir, folder, "videos")
        ldir = os.path.join(dataset_dir, folder, "labels")
        if not os.path.isdir(vdir):
            logger.warning(f"Not found: {vdir}")
            continue
        vpaths = find_videos(vdir)
        logger.info(f"Found {len(vpaths)} videos in {folder}/videos/")

        for vi, vp in enumerate(vpaths):
            vname = os.path.relpath(vp, dataset_dir)
            logger.info(f"  [{vi + 1}/{len(vpaths)}] {vname}")
            lpath = os.path.join(ldir, f"{Path(vp).stem}.csv")
            labels = load_labels(lpath)

            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for cs in range(0, total_frames, raw_per_clip):
                ce = cs + raw_per_clip - 1
                if ce >= total_frames:
                    break
                cs_sec, ce_sec = cs / fps, ce / fps

                cap.set(cv2.CAP_PROP_POS_FRAMES, cs)
                bgr, rgb = [], []
                for off in range(raw_per_clip):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if off % frame_step == 0:
                        bgr.append(frame)
                        rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(bgr) < num_frames:
                    break

                gt = (
                    1
                    if (
                        is_pos_folder
                        and clip_overlaps_any_label(cs_sec, ce_sec, labels)
                    )
                    else 0
                )
                yr = evaluate_clip_yolo(bgr, yolo_model, yolo_threshold)
                vr = evaluate_clip_vjepa(
                    rgb, encoder, classifier, positive_idx, device
                )

                yolo_pred = 1 if yr["yolo_frame_count"] > 0 else 0
                vjepa_pred = 1 if vr["vjepa_label"] == positive_idx else 0

                clips.append(
                    {
                        "video": vname,
                        "folder": folder,
                        "start_sec": round(cs_sec, 2),
                        "end_sec": round(ce_sec, 2),
                        "ground_truth": gt,
                        "yolo_pred": yolo_pred,
                        "yolo_max_conf": round(yr["yolo_max_conf"], 4),
                        "yolo_frame_ratio": round(yr["yolo_frame_ratio"], 4),
                        "vjepa_pred": vjepa_pred,
                        "vjepa_violent_conf": round(
                            vr["vjepa_violent_conf"], 4
                        ),
                    }
                )
            cap.release()
    return clips


def analyze_overlap(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all overlap statistics from per-clip predictions."""
    n = len(clips)
    if n == 0:
        return {"error": "no clips"}

    # Agreement matrix
    agree = {"both_neg": 0, "yolo_only": 0, "vjepa_only": 0, "both_pos": 0}
    # Per-quadrant ground-truth breakdown
    quad_gt = {q: {"actual_pos": 0, "actual_neg": 0} for q in agree}

    # Standalone confusion matrices
    yolo_cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    vjepa_cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for c in clips:
        yp, vp, gt = c["yolo_pred"], c["vjepa_pred"], c["ground_truth"]
        # agreement quadrant
        if yp == 0 and vp == 0:
            q = "both_neg"
        elif yp == 1 and vp == 0:
            q = "yolo_only"
        elif yp == 0 and vp == 1:
            q = "vjepa_only"
        else:
            q = "both_pos"
        agree[q] += 1
        quad_gt[q]["actual_pos" if gt == 1 else "actual_neg"] += 1

        # standalone CMs
        for pred, cm in [(yp, yolo_cm), (vp, vjepa_cm)]:
            if gt == 1 and pred == 1:
                cm["TP"] += 1
            elif gt == 0 and pred == 1:
                cm["FP"] += 1
            elif gt == 0 and pred == 0:
                cm["TN"] += 1
            else:
                cm["FN"] += 1

    agreement_rate = (agree["both_neg"] + agree["both_pos"]) / n
    kappa = cohens_kappa(agree, n)

    # Unique contributions
    yolo_unique_tp = quad_gt["yolo_only"]["actual_pos"]
    vjepa_unique_tp = quad_gt["vjepa_only"]["actual_pos"]
    yolo_unique_fp = quad_gt["yolo_only"]["actual_neg"]
    vjepa_unique_fp = quad_gt["vjepa_only"]["actual_neg"]
    overlap_tp = quad_gt["both_pos"]["actual_pos"]
    overlap_fp = quad_gt["both_pos"]["actual_neg"]

    yolo_metrics = compute_metrics(
        yolo_cm["TP"],
        yolo_cm["FP"],
        yolo_cm["TN"],
        yolo_cm["FN"],
    )
    vjepa_metrics = compute_metrics(
        vjepa_cm["TP"],
        vjepa_cm["FP"],
        vjepa_cm["TN"],
        vjepa_cm["FN"],
    )

    total_tp = yolo_cm["TP"] + vjepa_unique_tp  # union TP
    yolo_contribution = yolo_unique_tp / total_tp if total_tp else 0
    vjepa_contribution = vjepa_unique_tp / total_tp if total_tp else 0

    if kappa > 0.80:
        # High overlap — models are redundant, pick the better one
        if yolo_metrics["f1"] >= vjepa_metrics["f1"]:
            recommendation = "USE_YOLO_ONLY"
            reason = (
                f"High overlap (κ={kappa:.3f}). Models are redundant. "
                f"YOLO has better F1 ({yolo_metrics['f1']} vs {vjepa_metrics['f1']})."
            )
        else:
            recommendation = "USE_VJEPA_ONLY"
            reason = (
                f"High overlap (κ={kappa:.3f}). Models are redundant. "
                f"V-JEPA has better F1 ({vjepa_metrics['f1']} vs {yolo_metrics['f1']})."
            )
    elif vjepa_contribution < 0.05 and yolo_unique_fp <= vjepa_unique_fp:
        recommendation = "USE_YOLO_ONLY"
        reason = (
            f"V-JEPA contributes <5% unique TPs ({vjepa_unique_tp}) "
            f"and doesn't reduce FPs. Drop V-JEPA."
        )
    elif yolo_contribution < 0.05 and vjepa_unique_fp <= yolo_unique_fp:
        recommendation = "USE_VJEPA_ONLY"
        reason = (
            f"YOLO contributes <5% unique TPs ({yolo_unique_tp}) "
            f"and doesn't reduce FPs. Drop YOLO."
        )
    else:
        recommendation = "STACK_BOTH"
        reason = (
            f"Low overlap (κ={kappa:.3f}). "
            f"YOLO contributes {yolo_unique_tp} unique TPs, "
            f"V-JEPA contributes {vjepa_unique_tp} unique TPs. "
            f"Stacking can improve recall."
        )

    return {
        "total_clips": n,
        "agreement_matrix": agree,
        "agreement_matrix_gt_breakdown": quad_gt,
        "agreement_rate": round(agreement_rate, 4),
        "cohens_kappa": round(kappa, 4),
        "unique_contributions": {
            "yolo_unique_tp": yolo_unique_tp,
            "vjepa_unique_tp": vjepa_unique_tp,
            "yolo_unique_fp": yolo_unique_fp,
            "vjepa_unique_fp": vjepa_unique_fp,
            "overlap_tp": overlap_tp,
            "overlap_fp": overlap_fp,
        },
        "standalone_metrics": {
            "yolo": {"confusion_matrix": yolo_cm, "metrics": yolo_metrics},
            "vjepa": {"confusion_matrix": vjepa_cm, "metrics": vjepa_metrics},
        },
        "recommendation": recommendation,
        "recommendation_reason": reason,
    }


def print_report(analysis: Dict[str, Any]) -> None:
    a = analysis
    am = a["agreement_matrix"]
    uc = a["unique_contributions"]
    sm = a["standalone_metrics"]

    print("\n" + "=" * 78)
    print("  V-JEPA vs YOLO — PREDICTION OVERLAP ANALYSIS")
    print("=" * 78)
    print(f"  Total clips evaluated: {a['total_clips']}")
    print()

    # Agreement matrix
    print("  ┌─────────────────────────────────────────────┐")
    print("  │           AGREEMENT MATRIX                  │")
    print("  ├──────────────┬──────────────┬───────────────┤")
    print("  │              │  YOLO = 0    │  YOLO = 1     │")
    print("  ├──────────────┼──────────────┼───────────────┤")
    bn = am["both_neg"]
    yo = am["yolo_only"]
    vo = am["vjepa_only"]
    bp = am["both_pos"]
    print(f"  │  V-JEPA = 0  │  {bn:>6}      │  {yo:>6}       │")
    print(f"  │  V-JEPA = 1  │  {vo:>6}      │  {bp:>6}       │")
    print("  └──────────────┴──────────────┴───────────────┘")
    print()

    # Key statistics
    print(f"  Agreement Rate : {a['agreement_rate']:.1%}")
    print(f"  Cohen's Kappa  : {a['cohens_kappa']:.4f}")
    print()

    # Unique contributions
    print("  UNIQUE CONTRIBUTIONS:")
    print(
        f"    YOLO-unique TPs  : {uc['yolo_unique_tp']:>5}  (TPs only YOLO catches)"
    )
    print(
        f"    V-JEPA-unique TPs: {uc['vjepa_unique_tp']:>5}  (TPs only V-JEPA catches)"
    )
    print(f"    Overlap TPs      : {uc['overlap_tp']:>5}  (TPs both catch)")
    print(
        f"    YOLO-unique FPs  : {uc['yolo_unique_fp']:>5}  (FPs only YOLO makes)"
    )
    print(
        f"    V-JEPA-unique FPs: {uc['vjepa_unique_fp']:>5}  (FPs only V-JEPA makes)"
    )
    print(f"    Overlap FPs      : {uc['overlap_fp']:>5}  (FPs both make)")
    print()

    # Side-by-side metrics
    ym = sm["yolo"]["metrics"]
    vm = sm["vjepa"]["metrics"]
    yc = sm["yolo"]["confusion_matrix"]
    vc = sm["vjepa"]["confusion_matrix"]
    print(f"  {'Metric':<14} {'YOLO':>10} {'V-JEPA':>10}")
    print("  " + "-" * 36)
    print(f"  {'TP':<14} {yc['TP']:>10} {vc['TP']:>10}")
    print(f"  {'FP':<14} {yc['FP']:>10} {vc['FP']:>10}")
    print(f"  {'TN':<14} {yc['TN']:>10} {vc['TN']:>10}")
    print(f"  {'FN':<14} {yc['FN']:>10} {vc['FN']:>10}")
    print("  " + "-" * 36)
    for k in ("precision", "recall", "f1", "f2", "specificity", "accuracy"):
        print(f"  {k:<14} {ym[k]:>10.4f} {vm[k]:>10.4f}")
    print()

    # Recommendation
    print("─" * 78)
    rec = a["recommendation"]
    tag = {
        "USE_YOLO_ONLY": "🟢 USE YOLO ONLY",
        "USE_VJEPA_ONLY": "🟢 USE V-JEPA ONLY",
        "STACK_BOTH": "🔶 STACK BOTH MODELS",
    }[rec]
    print(f"  RECOMMENDATION: {tag}")
    print(f"  {a['recommendation_reason']}")
    print("=" * 78 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="V-JEPA vs YOLO prediction overlap analysis."
    )
    p.add_argument("--dataset", type=str, default="/tf/data/test-dataset")
    p.add_argument(
        "--encoder_weights",
        type=str,
        default="/tf/data/pretrained-models/vjepa2_1_vitG_384.pt",
    )
    p.add_argument(
        "--probe_weights",
        type=str,
        default="/tf/data/outputs/evals_2_1/vitG-384/weaponized_2cls/"
        "video_classification_frozen/weaponized-2cls-vitg16-384/best.pt",
    )
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--frame_step", type=int, default=4)
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument(
        "--yolo26-checkpoint",
        type=str,
        default="/tf/data/pretrained-models/yolo26m-weapon-det.pt",
    )
    p.add_argument("--yolo-threshold", type=float, default=0.41)
    p.add_argument("--output", type=str, default="overlap_report.json")
    p.add_argument("--output-csv", type=str, default="overlap_clips.csv")
    args = p.parse_args()

    # Validate
    for sub in ("violent/videos", "non-violent/videos"):
        if not os.path.isdir(os.path.join(args.dataset, sub)):
            raise SystemExit(
                f"Error: not found: {os.path.join(args.dataset, sub)}"
            )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info("Loading V-JEPA 2.1 encoder …")
    wb = os.path.basename(args.encoder_weights).lower()
    wp = wb.split("_")
    ap = wp[2] if len(wp) > 2 else ""
    if "vitg" in ap:
        from src.models.vision_transformer import (  # type: ignore
            vit_gigantic_xformers as vit_model,
        )
    elif "vitl" in ap:
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
    pd_ = torch.load(
        args.encoder_weights, map_location="cpu", weights_only=True
    )
    for key in ("ema_encoder", "target_encoder", "encoder"):
        if key in pd_:
            pd_ = pd_[key]
            break
    pd_ = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in pd_.items()
    }
    encoder.load_state_dict(pd_, strict=False)
    encoder.to(device).eval()

    logger.info("Loading attentive classifier probe …")
    from src.models.attentive_pooler import AttentiveClassifier  # type: ignore

    probe_dict = torch.load(
        args.probe_weights, map_location="cpu", weights_only=True
    )
    if "classifiers" in probe_dict:
        probe_dict = probe_dict["classifiers"][0]
    probe_dict = {k.replace("module.", ""): v for k, v in probe_dict.items()}

    num_classes = probe_dict["linear.weight"].shape[0]
    if num_classes == 2:
        positive_idx = 1
        class_labels = ["non-violent", "violent"]
    elif num_classes == 3:
        positive_idx = 2
        class_labels = ["non-violent", "fighting", "violent"]
    else:
        positive_idx = num_classes - 1
        class_labels = [f"class_{i}" for i in range(num_classes)]
    logger.info(
        f"Detected {num_classes}-class probe, positive_idx={positive_idx}"
    )

    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=16,
        depth=4,
        num_classes=num_classes,
    )
    classifier.load_state_dict(probe_dict, strict=False)
    classifier.to(device).eval()

    logger.info("Loading YOLOv26 weapon detector …")
    yolo_model = YOLO(args.yolo26_checkpoint)

    logger.info("Collecting per-clip predictions …")
    t0 = time.time()
    clips = collect_predictions(
        args.dataset,
        yolo_model,
        encoder,
        classifier,
        positive_idx,
        device,
        yolo_threshold=args.yolo_threshold,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
    )
    elapsed = time.time() - t0

    analysis = analyze_overlap(clips)

    print_report(analysis)

    report = {
        "config": {
            "dataset": os.path.abspath(args.dataset),
            "yolo_checkpoint": args.yolo26_checkpoint,
            "yolo_threshold": args.yolo_threshold,
            "encoder_weights": args.encoder_weights,
            "probe_weights": args.probe_weights,
            "num_classes": num_classes,
            "positive_idx": positive_idx,
            "class_labels": class_labels,
            "num_frames": args.num_frames,
            "frame_step": args.frame_step,
            "device": device,
            "elapsed_sec": round(elapsed, 1),
        },
        "analysis": analysis,
        "per_clip": clips,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report saved to: {args.output}")

    if clips:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=clips[0].keys())
            w.writeheader()
            w.writerows(clips)
        logger.info(f"Per-clip CSV saved to: {args.output_csv}")

    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
