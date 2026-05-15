import argparse
import csv
import glob
import json
import logging
import os
import pickle
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
    segments: List[Tuple[float, float]] = []
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
    vids: List[str] = []
    for ext in ("*.mp4", "*.avi"):
        vids.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(vids)


def run_yolo_on_clip(
    frames: List[np.ndarray],
    yolo_model: YOLO,
    threshold: float,
) -> Dict[str, float]:
    total = len(frames)
    det_count = 0
    max_conf = 0.0
    all_confs: List[float] = []
    total_boxes = 0
    max_box_area = 0.0
    frame_hits: List[bool] = []

    for frame in frames:
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        results = yolo_model.predict(frame, verbose=False, conf=threshold)
        frame_max = 0.0
        hit = False
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                hit = True
                total_boxes += len(r.boxes)
                c = float(r.boxes.conf.max())
                frame_max = max(frame_max, c)
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = box
                    area = float((x2 - x1) * (y2 - y1)) / frame_area
                    max_box_area = max(max_box_area, area)
        frame_hits.append(hit)
        if hit:
            det_count += 1
            max_conf = max(max_conf, frame_max)
            all_confs.append(frame_max)

    std_conf = float(np.std(all_confs)) if all_confs else 0.0
    frame_ratio = det_count / total if total else 0.0

    # Longest consecutive detection streak
    streak, max_streak = 0, 0
    for h in frame_hits:
        if h:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Confidence ramp (slope of per-frame max conf over time)
    conf_ramp = 0.0
    if len(all_confs) >= 3:
        x = np.arange(len(all_confs), dtype=np.float64)
        conf_ramp = float(np.polyfit(x, all_confs, 1)[0])

    return {
        "yolo_frame_count": det_count,
        "yolo_frame_ratio": frame_ratio,
        "yolo_max_conf": max_conf,
        "yolo_std_conf": std_conf,
        "yolo_total_detections": total_boxes,
        "yolo_detection_streak": max_streak,
        "yolo_max_box_area": max_box_area,
        "yolo_conf_ramp": conf_ramp,
    }


def run_vjepa_on_clip(
    frames,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    positive_idx: int,
    device: str,
) -> Dict[str, float]:
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

        # Prediction entropy (uncertainty measure)
        entropy = -float((probs * torch.log(probs + 1e-8)).sum())

        # Logit margin (gap between top-2 class logits)
        top2 = torch.topk(logits[0], k=min(2, logits.shape[1]))
        margin = (
            float(top2.values[0] - top2.values[1])
            if logits.shape[1] >= 2
            else 0.0
        )

    return {
        "vjepa_label": float(label),
        "vjepa_conf": float(probs[label].item()),
        "vjepa_violent_conf": float(probs[positive_idx].item()),
        "vjepa_entropy": entropy,
        "vjepa_logit_margin": margin,
    }


def collect_clip_features(
    dataset_dir: str,
    yolo_model: YOLO,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    positive_idx: int,
    feature_columns: List[str],
    device: str,
    *,
    yolo_threshold: float,
    num_frames: int,
    frame_step: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Extract features for every clip. Returns X, y, metadata."""
    raw_per_clip = num_frames * frame_step
    all_features: List[List[float]] = []
    all_labels: List[int] = []
    metadata: List[Dict[str, Any]] = []

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
                    if is_pos_folder
                    and clip_overlaps_any_label(cs_sec, ce_sec, labels)
                    else 0
                )

                yr = run_yolo_on_clip(bgr, yolo_model, yolo_threshold)
                vr = run_vjepa_on_clip(
                    rgb, encoder, classifier, positive_idx, device
                )

                merged = {**yr, **vr}
                # Cross-model agreement
                yolo_says = 1.0 if merged["yolo_frame_ratio"] > 0 else 0.0
                vjepa_says = 1.0 if merged["vjepa_violent_conf"] > 0.5 else 0.0
                merged["yolo_x_vjepa_agreement"] = yolo_says * vjepa_says

                feat_vec = [merged.get(c, 0.0) for c in feature_columns]

                all_features.append(feat_vec)
                all_labels.append(gt)
                metadata.append(
                    {
                        "video": vname,
                        "folder": folder,
                        "start_sec": round(cs_sec, 2),
                        "end_sec": round(ce_sec, 2),
                        "ground_truth": gt,
                        **{
                            c: round(merged.get(c, 0.0), 4)
                            for c in feature_columns
                        },
                    }
                )

            cap.release()

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    return X, y, metadata


def compute_eval_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = tp + fp + tn + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "specificity": round(spec, 4),
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
    }


def print_results(
    meta_metrics: Dict[str, Any],
    yolo_metrics: Dict[str, Any],
    vjepa_metrics: Dict[str, Any],
    model_name: str,
) -> None:
    print("\n" + "=" * 78)
    print("  META-MODEL INFERENCE RESULTS (Test Dataset)")
    print("=" * 78)

    header = (
        f"  {'Model':<22}"
        f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'F2':>6} {'Spec':>6}"
    )
    print(header)
    print("  " + "-" * 74)

    for label, m in [
        (f"Meta ({model_name})", meta_metrics),
        ("YOLO-only", yolo_metrics),
        ("V-JEPA-only", vjepa_metrics),
    ]:
        print(
            f"  {label:<22}"
            f"{m['tp']:>5} {m['fp']:>5} {m['tn']:>5} {m['fn']:>5}  "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['f2']:>6.3f} "
            f"{m['specificity']:>6.3f}"
        )

    meta_f1 = meta_metrics["f1"]
    yolo_f1 = yolo_metrics["f1"]
    vjepa_f1 = vjepa_metrics["f1"]
    best_single = max(yolo_f1, vjepa_f1)
    improvement = meta_f1 - best_single

    print("-" * 78)
    if improvement > 0:
        print(
            f"  Meta-model improves F1 by {improvement:.4f} "
            f"over best single model"
        )
    elif improvement == 0:
        print("  Meta-model matches best single model")
    else:
        print(
            f"  Meta-model is {abs(improvement):.4f} F1 worse "
            f"than best single model"
        )

    fp_saved = (
        min(yolo_metrics["fp"], vjepa_metrics["fp"]) - meta_metrics["fp"]
    )
    if fp_saved > 0:
        print(f"  Meta-model saves {fp_saved} false positive(s)")

    print("=" * 78 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Test a trained meta-learner on a new dataset."
    )
    p.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="Path to test dataset (violent/ + non-violent/ subdirs)",
    )
    p.add_argument(
        "--meta-model",
        type=str,
        default="meta_model.pkl",
        help="Path to saved meta_model.pkl from meta-model-stacking.py",
    )
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
    p.add_argument("--output", type=str, default="inference_report.json")
    p.add_argument("--output-csv", type=str, default="inference_clips.csv")
    args = p.parse_args()

    # Load meta-model
    logger.info(f"Loading meta-model from: {args.meta_model}")
    with open(args.meta_model, "rb") as f:
        saved = pickle.load(f)

    meta_clf = saved["model"]
    model_name = saved["model_name"]
    feature_columns = saved["feature_columns"]
    train_metrics = saved["train_metrics"]
    logger.info(
        f"Loaded {model_name} — "
        f"train F1={train_metrics['f1']}, "
        f"{len(feature_columns)} features: {feature_columns}"
    )

    # Validate test dataset
    for sub in ("violent/videos", "non-violent/videos"):
        if not os.path.isdir(os.path.join(args.test_dataset, sub)):
            raise SystemExit(
                f"Error: not found: {os.path.join(args.test_dataset, sub)}"
            )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load V-JEPA encoder
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

    # Load classifier probe
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
    elif num_classes == 3:
        positive_idx = 2
    else:
        positive_idx = num_classes - 1
    logger.info(f"{num_classes}-class probe, positive_idx={positive_idx}")

    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=16,
        depth=4,
        num_classes=num_classes,
    )
    classifier.load_state_dict(probe_dict, strict=False)
    classifier.to(device).eval()

    # Load YOLO
    logger.info("Loading YOLOv26 weapon detector …")
    yolo_model = YOLO(args.yolo26_checkpoint)

    # Extract features from test dataset
    logger.info("Extracting features from test dataset …")
    t0 = time.time()
    X, y, metadata = collect_clip_features(
        args.test_dataset,
        yolo_model,
        encoder,
        classifier,
        positive_idx,
        feature_columns,
        device,
        yolo_threshold=args.yolo_threshold,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
    )
    elapsed = time.time() - t0
    logger.info(
        f"Extracted {X.shape[0]} clips × {X.shape[1]} features "
        f"in {elapsed:.1f}s"
    )

    if len(y) == 0:
        raise SystemExit("Error: no clips extracted from test dataset.")

    # Run meta-model predictions
    logger.info("Running meta-model predictions …")
    meta_pred = meta_clf.predict(X)
    meta_proba = None
    if hasattr(meta_clf, "predict_proba"):
        meta_proba = meta_clf.predict_proba(X)

    meta_metrics = compute_eval_metrics(y, meta_pred)

    # Baselines
    yolo_fc_idx = feature_columns.index("yolo_frame_ratio")
    yolo_pred = (X[:, yolo_fc_idx] > 0).astype(int)
    yolo_metrics = compute_eval_metrics(y, yolo_pred)

    vjepa_vc_idx = feature_columns.index("vjepa_violent_conf")
    vjepa_pred = (X[:, vjepa_vc_idx] > 0.5).astype(int)
    vjepa_metrics = compute_eval_metrics(y, vjepa_pred)

    print_results(meta_metrics, yolo_metrics, vjepa_metrics, model_name)

    # Annotate metadata with predictions
    for i, clip in enumerate(metadata):
        clip["meta_pred"] = int(meta_pred[i])
        clip["yolo_pred"] = int(yolo_pred[i])
        clip["vjepa_pred"] = int(vjepa_pred[i])
        if meta_proba is not None:
            clip["meta_proba_positive"] = round(float(meta_proba[i, 1]), 4)

        gt = clip["ground_truth"]
        pred = int(meta_pred[i])
        if gt == 1 and pred == 1:
            clip["meta_outcome"] = "TP"
        elif gt == 0 and pred == 1:
            clip["meta_outcome"] = "FP"
        elif gt == 0 and pred == 0:
            clip["meta_outcome"] = "TN"
        else:
            clip["meta_outcome"] = "FN"

    # Save reports
    report = {
        "config": {
            "test_dataset": os.path.abspath(args.test_dataset),
            "meta_model_path": args.meta_model,
            "meta_model_name": model_name,
            "feature_columns": feature_columns,
            "yolo_threshold": args.yolo_threshold,
            "device": device,
            "elapsed_sec": round(elapsed, 1),
        },
        "dataset_stats": {
            "total_clips": len(y),
            "positive_clips": int(y.sum()),
            "negative_clips": int(len(y) - y.sum()),
        },
        "results": {
            "meta_model": meta_metrics,
            "yolo_only": yolo_metrics,
            "vjepa_only": vjepa_metrics,
        },
        "per_clip": metadata,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report saved to: {args.output}")

    if metadata:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=metadata[0].keys())
            w.writeheader()
            w.writerows(metadata)
        logger.info(f"Per-clip CSV saved to: {args.output_csv}")

    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
