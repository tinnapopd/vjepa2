import argparse
import csv
import json
import logging
import os
import pickle
import time
import warnings
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
from ultralytics import YOLO
from xgboost import XGBClassifier

from src.models.attentive_pooler import AttentiveClassifier  # type: ignore
import src.datasets.utils.video.transforms as video_transforms  # type: ignore
import src.datasets.utils.video.volume_transforms as volume_transforms  # type: ignore

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


FEATURE_COLUMNS = [
    "yolo_frame_ratio",       # fraction of frames where object is detected
    "yolo_max_conf",          # maximum confidence of the object detection
    "yolo_std_conf",          # standard deviation of the object detection confidence
    "yolo_total_detections",  # total bounding box count across all frames
    "yolo_detection_streak",  # longest consecutive run of frames with detections
    "yolo_max_box_area",      # largest detected weapon area (normalized by frame)
    "yolo_conf_ramp",         # confidence trend over time (slope)
    "vjepa_violent_conf",     # confidence of violent class
    "vjepa_conf",             # confidence of the predicted class
    "vjepa_entropy",          # prediction entropy (uncertainty)
    "vjepa_logit_margin",     # gap between top-2 class logits
    "yolo_x_vjepa_agreement", # both models agree on violence
]


def run_yolo_on_clip(
    frames: List[np.ndarray],
    yolo_model: YOLO,
    threshold: float,
) -> Dict[str, float]:
    total = len(frames)
    det_count = 0
    max_conf = 0.0
    all_confs: list[float] = []
    total_boxes = 0
    max_box_area = 0.0
    frame_hits: list[bool] = []

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


def parse_dataset_csv(csv_path: str) -> List[Tuple[str, int]]:
    entries: List[Tuple[str, int]] = []
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                parts = line.rsplit(",", 1)
            else:
                parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            video_path = parts[0].strip()
            try:
                label = int(parts[1].strip())
            except ValueError:
                continue
            if not os.path.isabs(video_path):
                video_path = os.path.normpath(
                    os.path.join(csv_dir, video_path)
                )
            entries.append((video_path, label))
    return entries


def collect_clip_features_from_csv(
    csv_path: str,
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
    entries = parse_dataset_csv(csv_path)
    if not entries:
        raise SystemExit(f"Error: no entries found in {csv_path}")

    logger.info(f"Loaded {len(entries)} videos from {csv_path}")

    raw_per_clip = num_frames * frame_step
    all_features = []
    all_labels = []
    metadata = []

    feat_cols = list(FEATURE_COLUMNS)

    for vi, (vp, video_label) in enumerate(entries):
        gt = 1 if video_label == positive_idx else 0

        logger.info(f"  [{vi + 1}/{len(entries)}] {vp} (label={video_label})")

        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            logger.warning(f"Cannot open: {vp}")
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

            yr = run_yolo_on_clip(bgr, yolo_model, yolo_threshold)
            vr = run_vjepa_on_clip(
                rgb,
                encoder,
                classifier,
                positive_idx,
                device,
            )

            merged = {**yr, **vr}
            # Cross-model agreement
            yolo_says = 1.0 if merged["yolo_frame_ratio"] > 0 else 0.0
            vjepa_says = 1.0 if merged["vjepa_violent_conf"] > 0.5 else 0.0
            merged["yolo_x_vjepa_agreement"] = yolo_says * vjepa_says

            feat_vec = [merged.get(c, 0.0) for c in feat_cols]

            all_features.append(feat_vec)
            all_labels.append(gt)
            metadata.append(
                {
                    "video": vp,
                    "video_label": video_label,
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


def compute_eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
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
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    results = {}

    yolo_pred = (X[:, FEATURE_COLUMNS.index("yolo_frame_ratio")] > 0).astype(
        int
    )
    results["YOLO_only"] = {
        "metrics": compute_eval_metrics(y, yolo_pred),
        "predictions": yolo_pred.tolist(),
    }

    vjepa_pred = (
        X[:, FEATURE_COLUMNS.index("vjepa_violent_conf")] > 0.5
    ).astype(int)
    results["VJEPA_only"] = {
        "metrics": compute_eval_metrics(y, vjepa_pred),
        "predictions": vjepa_pred.tolist(),
    }

    for name, model in meta_models.items():
        logger.info(f"Training meta-model: {name} ({n_splits}-fold OOF)")
        try:
            oof_preds = cross_val_predict(
                model, X, y, cv=skf, method="predict"
            )
            metrics = compute_eval_metrics(y, oof_preds)

            model.fit(X, y)

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


def evaluate_on_val(
    results: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate all trained meta-models on a held-out validation set."""
    yolo_pred = (X_val[:, FEATURE_COLUMNS.index("yolo_frame_ratio")] > 0).astype(int)
    results["YOLO_only"]["val_metrics"] = compute_eval_metrics(y_val, yolo_pred)

    vjepa_pred = (X_val[:, FEATURE_COLUMNS.index("vjepa_violent_conf")] > 0.5).astype(int)
    results["VJEPA_only"]["val_metrics"] = compute_eval_metrics(y_val, vjepa_pred)

    for name, entry in results.items():
        if name in ("YOLO_only", "VJEPA_only"):
            continue
        if "model_instance" not in entry:
            continue
        model = entry["model_instance"]
        try:
            val_preds = model.predict(X_val)
            entry["val_metrics"] = compute_eval_metrics(y_val, val_preds)
            vm = entry["val_metrics"]
            logger.info(
                f"  {name} val: F1={vm['f1']:.4f}  "
                f"Prec={vm['precision']:.4f}  Rec={vm['recall']:.4f}"
            )
        except Exception as e:
            logger.error(f"  {name} val failed: {e}")
            entry["val_metrics"] = {"error": str(e)}
    return results


def _print_section(
    results: Dict[str, Any],
    metric_key: str,
    title: str,
    sort_key: str,
) -> str:
    """Print one section of the comparison table. Returns best model name."""
    print(f"\n  {title}")
    print("  " + "-" * 78)
    header = (
        f"  {'Model':<22}"
        f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'F2':>6} {'Spec':>6}"
    )
    print(header)
    print("  " + "-" * 78)

    has_metric = [
        k for k in results
        if isinstance(results[k].get(metric_key), dict)
        and "f1" in results[k][metric_key]
    ]
    sorted_models = sorted(
        has_metric,
        key=lambda k: results[k][metric_key].get("f1", 0),
        reverse=True,
    )
    best_name = sorted_models[0] if sorted_models else ""

    for name in sorted_models:
        m = results[name][metric_key]
        marker = " ★" if name == best_name else ""
        print(
            f"  {name:<22}"
            f"{m['tp']:>5} {m['fp']:>5} {m['tn']:>5} {m['fn']:>5}  "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['f2']:>6.3f} "
            f"{m['specificity']:>6.3f}{marker}"
        )
    return best_name


def print_comparison_table(
    results: Dict[str, Any],
    feature_cols: List[str],
    has_val: bool = False,
) -> None:
    print("\n" + "=" * 82)
    print("  META-MODEL STACKING RESULTS")
    print("=" * 82)

    _print_section(results, "metrics", "Train (OOF)", "f1")

    best_name = ""
    if has_val:
        best_name = _print_section(
            results, "val_metrics", "Validation", "f1"
        )
    else:
        has_m = [
            k for k in results
            if isinstance(results[k].get("metrics"), dict)
            and "f1" in results[k]["metrics"]
        ]
        if has_m:
            best_name = max(
                has_m, key=lambda k: results[k]["metrics"].get("f1", 0)
            )

    if best_name and "feature_importances" in results.get(best_name, {}):
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-csv",
        type=str,
        default="/tf/data/dataset-2classes/train.csv",
        help="Path to training CSV: '<video_path> <label>' per line",
    )
    p.add_argument(
        "--val-csv",
        type=str,
        default="/tf/data/dataset-2classes/val.csv",
        help="Path to validation CSV (same format). Best model selected by val F1.",
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

    if not os.path.isfile(args.dataset_csv):
        raise SystemExit(f"Error: CSV not found: {args.dataset_csv}")

    has_val = os.path.isfile(args.val_csv)
    if not has_val:
        logger.warning(
            f"Val CSV not found: {args.val_csv} — "
            "will select best model by OOF F1 only"
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

    # Load YOLO
    logger.info("Loading YOLOv26 weapon detector …")
    yolo_model = YOLO(args.yolo26_checkpoint)

    # Collect features from all clips
    logger.info("Collecting per-clip features …")
    t0 = time.time()
    X, y, metadata = collect_clip_features_from_csv(
        args.dataset_csv,
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

    # Build feature column names
    feat_cols = list(FEATURE_COLUMNS)

    # Train and evaluate meta-models
    logger.info("Training meta-models with OOF cross-validation …")
    t1 = time.time()
    results = train_and_evaluate_stacking(
        X,
        y,
        n_splits=args.n_folds,
    )
    train_time = time.time() - t1

    # Evaluate on validation set
    val_time = 0.0
    if has_val:
        logger.info("Collecting val-set features …")
        tv = time.time()
        X_val, y_val, val_metadata = collect_clip_features_from_csv(
            args.val_csv,
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
        logger.info("Evaluating meta-models on validation set …")
        results = evaluate_on_val(results, X_val, y_val)
        val_time = time.time() - tv

    # Print results
    print_comparison_table(results, feat_cols, has_val=has_val)

    # Save best model
    meta_only = {
        k: v
        for k, v in results.items()
        if "metrics" in v and "model_instance" in v
    }
    if meta_only:
        select_key = "val_metrics" if has_val else "metrics"
        # Filter to models that have the selection metric
        candidates = {
            k: v for k, v in meta_only.items()
            if isinstance(v.get(select_key), dict) and "f1" in v[select_key]
        }
        if not candidates:
            candidates = meta_only
            select_key = "metrics"
        best_name = max(
            candidates, key=lambda k: candidates[k][select_key].get("f1", 0)
        )
        best_model = candidates[best_name]["model_instance"]
        save_payload: Dict[str, Any] = {
            "model": best_model,
            "model_name": best_name,
            "feature_columns": feat_cols,
            "train_metrics": candidates[best_name]["metrics"],
        }
        if has_val and "val_metrics" in candidates[best_name]:
            save_payload["val_metrics"] = candidates[best_name]["val_metrics"]
        with open(args.save_model, "wb") as f:
            pickle.dump(save_payload, f)
        logger.info(
            f"Best meta-model ({best_name}) saved to: {args.save_model}"
        )

    # Save reports
    serializable = {}
    for k, v in results.items():
        entry = {kk: vv for kk, vv in v.items() if kk != "model_instance"}
        serializable[k] = entry

    report: Dict[str, Any] = {
        "config": {
            "dataset_csv": os.path.abspath(args.dataset_csv),
            "val_csv": os.path.abspath(args.val_csv) if has_val else None,
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
            "val_time_sec": round(val_time, 1),
        },
        "dataset_stats": {
            "train_clips": len(y),
            "train_positive": int(y.sum()),
            "train_negative": int(len(y) - y.sum()),
            "n_features": X.shape[1],
        },
        "results": serializable,
    }
    if has_val:
        report["dataset_stats"]["val_clips"] = len(y_val)
        report["dataset_stats"]["val_positive"] = int(y_val.sum())
        report["dataset_stats"]["val_negative"] = int(len(y_val) - y_val.sum())

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
        f"Done — collection: {collect_time:.1f}s, "
        f"training: {train_time:.1f}s, val: {val_time:.1f}s"
    )


if __name__ == "__main__":
    main()
