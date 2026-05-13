"""Meta-Model Stacking: V-JEPA + YOLO Combined Prediction.

Uses model stacking (inspired by the TDS article on model stacking) to train
a meta-classifier on top of V-JEPA and YOLO predictions.  Out-of-fold (OOF)
predictions prevent data leakage when training the meta-model.

Pipeline:
  1. Run both V-JEPA and YOLO on every clip → extract feature vector per clip.
  2. Train meta-models via K-Fold cross-validation (OOF predictions).
  3. Evaluate each meta-model and select the best one.
  4. Report metrics and save the trained meta-model.

Usage:
    python meta-model-stacking.py \
        --dataset /tf/data/test-dataset \
        --encoder_weights pretrained-models/vjepa2_1_vitG_384.pt \
        --probe_weights trained-probes/vitG-probe.pt \
        --yolo26-checkpoint trained-models/yolo26m-weapon-det.pt
"""

from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ultralytics import YOLO

import src.datasets.utils.video.transforms as video_transforms  # type: ignore
import src.datasets.utils.video.volume_transforms as volume_transforms  # type: ignore

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Shared helpers (kept minimal, mirrors pred-overlapping.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-clip feature extraction
# ---------------------------------------------------------------------------


def run_yolo_on_clip(
    frames: List[np.ndarray],
    yolo_model: YOLO,
    threshold: float,
) -> Dict[str, float]:
    """Run YOLO on every frame, return aggregated features."""
    total = len(frames)
    det_count = 0
    max_conf = 0.0
    conf_sum = 0.0
    all_confs: List[float] = []

    for frame in frames:
        results = yolo_model.predict(frame, verbose=False, conf=threshold)
        frame_max = 0.0
        hit = False
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                hit = True
                c = float(r.boxes.conf.max())
                frame_max = max(frame_max, c)
        if hit:
            det_count += 1
            max_conf = max(max_conf, frame_max)
            conf_sum += frame_max
            all_confs.append(frame_max)

    avg_conf = conf_sum / det_count if det_count else 0.0
    std_conf = float(np.std(all_confs)) if all_confs else 0.0
    frame_ratio = det_count / total if total else 0.0

    return {
        "yolo_frame_count": det_count,
        "yolo_frame_ratio": frame_ratio,
        "yolo_max_conf": max_conf,
        "yolo_avg_conf": avg_conf,
        "yolo_std_conf": std_conf,
    }


def run_vjepa_on_clip(
    frames,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    positive_idx: int,
    device: str,
) -> Dict[str, float]:
    """Run V-JEPA on a clip, return classification features."""
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

    # Return all class probabilities as features
    result: Dict[str, float] = {
        "vjepa_label": float(label),
        "vjepa_conf": float(probs[label].item()),
        "vjepa_violent_conf": float(probs[positive_idx].item()),
    }
    # Add all per-class probabilities as separate features
    for i in range(probs.shape[0]):
        result[f"vjepa_prob_cls{i}"] = float(probs[i].item())
    return result


# ---------------------------------------------------------------------------
# Dataset collection: extract features for every clip
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "yolo_frame_count",
    "yolo_frame_ratio",
    "yolo_max_conf",
    "yolo_avg_conf",
    "yolo_std_conf",
    "vjepa_violent_conf",
    "vjepa_conf",
]


def collect_clip_features(
    dataset_dir: str,
    yolo_model: YOLO,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    positive_idx: int,
    num_classes: int,
    device: str,
    *,
    yolo_threshold: float,
    num_frames: int,
    frame_step: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Collect stacking features for every clip in the dataset.

    Returns:
        X: np.ndarray of shape (n_clips, n_features)
        y: np.ndarray of shape (n_clips,) — ground-truth labels (0/1)
        metadata: list of per-clip dicts for traceability
    """
    raw_per_clip = num_frames * frame_step
    all_features: List[List[float]] = []
    all_labels: List[int] = []
    metadata: List[Dict[str, Any]] = []

    # Build dynamic feature list including per-class vjepa probs
    feat_cols = list(FEATURE_COLUMNS)
    for i in range(num_classes):
        col = f"vjepa_prob_cls{i}"
        if col not in feat_cols:
            feat_cols.append(col)

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

                # Ground truth
                gt = (
                    1
                    if is_pos_folder
                    and clip_overlaps_any_label(cs_sec, ce_sec, labels)
                    else 0
                )

                # Run both models
                yr = run_yolo_on_clip(bgr, yolo_model, yolo_threshold)
                vr = run_vjepa_on_clip(
                    rgb, encoder, classifier, positive_idx, device
                )

                # Build feature vector
                merged = {**yr, **vr}
                feat_vec = [merged.get(c, 0.0) for c in feat_cols]

                all_features.append(feat_vec)
                all_labels.append(gt)
                metadata.append(
                    {
                        "video": vname,
                        "folder": folder,
                        "start_sec": round(cs_sec, 2),
                        "end_sec": round(ce_sec, 2),
                        "ground_truth": gt,
                        **{c: round(merged.get(c, 0.0), 4) for c in feat_cols},
                    }
                )

            cap.release()

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    logger.info(
        f"Collected {len(y)} clips — {int(y.sum())} positive, "
        f"{len(y) - int(y.sum())} negative, {X.shape[1]} features"
    )
    return X, y, metadata


# ---------------------------------------------------------------------------
# Meta-model training & evaluation (OOF stacking)
# ---------------------------------------------------------------------------


def compute_eval_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
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


def train_and_evaluate_stacking(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train multiple meta-models using OOF predictions, return results."""
    meta_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=random_state,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        ),
    }

    # Try importing XGBoost (optional)
    try:
        from xgboost import XGBClassifier

        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        scale = neg_count / pos_count if pos_count > 0 else 1.0
        meta_models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    except ImportError:
        logger.info("XGBoost not installed — skipping.")

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    results: Dict[str, Any] = {}

    # --- Baselines: YOLO-only and V-JEPA-only ---
    # YOLO-only baseline: positive if any frame has detection
    yolo_pred = (X[:, FEATURE_COLUMNS.index("yolo_frame_count")] > 0).astype(
        int
    )
    results["YOLO_only"] = {
        "metrics": compute_eval_metrics(y, yolo_pred),
        "predictions": yolo_pred.tolist(),
    }

    # V-JEPA-only baseline: positive if violent_conf > 0.5
    vjepa_pred = (
        X[:, FEATURE_COLUMNS.index("vjepa_violent_conf")] > 0.5
    ).astype(int)
    results["VJEPA_only"] = {
        "metrics": compute_eval_metrics(y, vjepa_pred),
        "predictions": vjepa_pred.tolist(),
    }

    # --- OOF meta-model training ---
    for name, model in meta_models.items():
        logger.info(f"Training meta-model: {name} ({n_splits}-fold OOF)")
        try:
            oof_preds = cross_val_predict(
                model, X, y, cv=skf, method="predict"
            )
            metrics = compute_eval_metrics(y, oof_preds)

            # Fit final model on all data for saving
            model.fit(X, y)

            # Feature importances (if available)
            importances = None
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_.tolist()
            elif hasattr(model, "coef_"):
                importances = model.coef_[0].tolist()

            results[name] = {
                "metrics": metrics,
                "predictions": oof_preds.tolist(),
                "feature_importances": importances,
                "model_instance": model,
            }
            logger.info(
                f"  {name}: F1={metrics['f1']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"Rec={metrics['recall']:.4f}  "
                f"FP={metrics['fp']}  FN={metrics['fn']}"
            )
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            results[name] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_comparison_table(
    results: Dict[str, Any], feature_cols: List[str]
) -> None:
    print("\n" + "=" * 82)
    print("  META-MODEL STACKING RESULTS")
    print("=" * 82)

    header = (
        f"  {'Model':<22}"
        f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'F2':>6} {'Spec':>6}"
    )
    print(header)
    print("  " + "-" * 78)

    sorted_models = sorted(
        results.keys(),
        key=lambda k: results[k].get("metrics", {}).get("f1", 0),
        reverse=True,
    )

    best_name = sorted_models[0] if sorted_models else ""

    for name in sorted_models:
        entry = results[name]
        if "error" in entry:
            print(f"  {name:<22} ERROR: {entry['error']}")
            continue
        m = entry["metrics"]
        marker = " ★" if name == best_name else ""
        print(
            f"  {name:<22}"
            f"{m['tp']:>5} {m['fp']:>5} {m['tn']:>5} {m['fn']:>5}  "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['f2']:>6.3f} "
            f"{m['specificity']:>6.3f}{marker}"
        )

    # Feature importances for the best model
    if best_name and "feature_importances" in results[best_name]:
        imps = results[best_name]["feature_importances"]
        if imps is not None:
            print(f"\n  Feature importances ({best_name}):")
            pairs = sorted(
                zip(feature_cols, imps), key=lambda x: abs(x[1]), reverse=True
            )
            for fname, imp in pairs:
                bar = "█" * int(abs(imp) / max(abs(i) for _, i in pairs) * 20)
                print(f"    {fname:<24} {imp:>8.4f}  {bar}")

    print("=" * 82 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Meta-model stacking: V-JEPA + YOLO."
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
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--output", type=str, default="stacking_report.json")
    p.add_argument("--output-csv", type=str, default="stacking_clips.csv")
    p.add_argument(
        "--save-model",
        type=str,
        default="meta_model.pkl",
        help="Path to save the best meta-model (pickle)",
    )
    args = p.parse_args()

    # Validate dataset
    for sub in ("violent/videos", "non-violent/videos"):
        if not os.path.isdir(os.path.join(args.dataset, sub)):
            raise SystemExit(
                f"Error: not found: {os.path.join(args.dataset, sub)}"
            )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # --- Load V-JEPA encoder ---
    logger.info("Loading V-JEPA 2.1 encoder …")
    wb = os.path.basename(args.encoder_weights).lower()
    wp = wb.split("_")
    ap = wp[2] if len(wp) > 2 else ""
    if "vitg" in ap:
        from src.models.vision_transformer import (
            vit_gigantic_xformers as vit_model,
        )
    elif "vitl" in ap:
        from src.models.vision_transformer import vit_large as vit_model
    else:
        from src.models.vision_transformer import vit_base as vit_model

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

    # --- Load classifier probe ---
    logger.info("Loading attentive classifier probe …")
    from src.models.attentive_pooler import AttentiveClassifier

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
    logger.info(f"{num_classes}-class probe, positive_idx={positive_idx}")

    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=16,
        depth=4,
        num_classes=num_classes,
    )
    classifier.load_state_dict(probe_dict, strict=False)
    classifier.to(device).eval()

    # --- Load YOLO ---
    logger.info("Loading YOLOv26 weapon detector …")
    yolo_model = YOLO(args.yolo26_checkpoint)

    # --- Step 1: Collect features from all clips ---
    logger.info("Collecting per-clip features …")
    t0 = time.time()
    X, y, metadata = collect_clip_features(
        args.dataset,
        yolo_model,
        encoder,
        classifier,
        positive_idx,
        num_classes,
        device,
        yolo_threshold=args.yolo_threshold,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
    )
    collect_time = time.time() - t0
    logger.info(f"Feature collection done in {collect_time:.1f}s")

    # Build feature column names (dynamic based on num_classes)
    feat_cols = list(FEATURE_COLUMNS)
    for i in range(num_classes):
        col = f"vjepa_prob_cls{i}"
        if col not in feat_cols:
            feat_cols.append(col)

    # --- Step 2: Train & evaluate meta-models ---
    logger.info("Training meta-models with OOF cross-validation …")
    t1 = time.time()
    results = train_and_evaluate_stacking(
        X,
        y,
        n_splits=args.n_folds,
    )
    train_time = time.time() - t1

    # --- Step 3: Print results ---
    print_comparison_table(results, feat_cols)

    # --- Step 4: Save best model ---
    meta_only = {
        k: v
        for k, v in results.items()
        if "metrics" in v and "model_instance" in v
    }
    if meta_only:
        best_name = max(meta_only, key=lambda k: meta_only[k]["metrics"]["f1"])
        best_model = meta_only[best_name]["model_instance"]
        with open(args.save_model, "wb") as f:
            pickle.dump(
                {
                    "model": best_model,
                    "model_name": best_name,
                    "feature_columns": feat_cols,
                    "metrics": meta_only[best_name]["metrics"],
                },
                f,
            )
        logger.info(
            f"Best meta-model ({best_name}) saved to: {args.save_model}"
        )

    # --- Step 5: Save reports ---
    # Strip non-serializable model instances
    serializable = {}
    for k, v in results.items():
        entry = {kk: vv for kk, vv in v.items() if kk != "model_instance"}
        serializable[k] = entry

    report = {
        "config": {
            "dataset": os.path.abspath(args.dataset),
            "yolo_threshold": args.yolo_threshold,
            "encoder_weights": args.encoder_weights,
            "probe_weights": args.probe_weights,
            "num_classes": num_classes,
            "positive_idx": positive_idx,
            "class_labels": class_labels,
            "num_frames": args.num_frames,
            "frame_step": args.frame_step,
            "n_folds": args.n_folds,
            "feature_columns": feat_cols,
            "device": device,
            "collect_time_sec": round(collect_time, 1),
            "train_time_sec": round(train_time, 1),
        },
        "dataset_stats": {
            "total_clips": len(y),
            "positive_clips": int(y.sum()),
            "negative_clips": int(len(y) - y.sum()),
            "n_features": X.shape[1],
        },
        "results": serializable,
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

    logger.info(
        f"Done — collection: {collect_time:.1f}s, training: {train_time:.1f}s"
    )


if __name__ == "__main__":
    main()
