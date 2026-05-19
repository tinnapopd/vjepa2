"""Meta-model training using V-JEPA + YOLO-CLS embeddings.

Trains a meta-learner on concatenated raw embeddings:
  [V-JEPA pooled embedding ∥ YOLO-CLS penultimate-layer embedding]

Optionally reduces dimensionality with PCA before training.

Usage:
    python meta_training.py \\
        --dataset-csv /tf/data/dataset-2classes/train.csv \\
        --val-csv /tf/data/dataset-2classes/val.csv \\
        --cls-checkpoint trained-models/yolo26m-violence-cls/best.pt \\
        --pca-dim 128
"""

import argparse
import json
import logging
import os
import pickle
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from meta_common import (  # type: ignore
    PipelineModels,
    PipelineStrategy,
    add_shared_model_args,
    add_strategy_arg,
    collect_clip_features,
    compute_eval_metrics,
    load_pipeline_models,
    write_clips_csv,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_dataset_csv(csv_path: str) -> List[Tuple[str, int]]:
    entries = []
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


def collect_embeddings_from_csv(
    csv_path: str,
    models: PipelineModels,
    device: str,
    num_frames: int,
    frame_step: int,
    human_threshold: float = 0.3,
    weapon_threshold: float = 0.41,
    strategy: Optional[PipelineStrategy] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    entries = parse_dataset_csv(csv_path)
    if not entries:
        raise SystemExit(f"Error: no entries found in {csv_path}")

    logger.info(f"Loaded {len(entries)} videos from {csv_path}")

    def video_entries() -> Iterator[Tuple[str, Dict[str, Any]]]:
        for vi, (vp, video_label) in enumerate(entries):
            logger.info(
                f"  [{vi + 1}/{len(entries)}] {vp} (label={video_label})"
            )
            yield vp, {"video_label": video_label}

    def label_fn(ctx: Dict[str, Any], cs: float, ce: float) -> int:
        return 1 if ctx["video_label"] == models.positive_idx else 0

    def metadata_fn(
        vp: str, ctx: Dict[str, Any], cs: float, ce: float, gt: int
    ) -> Dict[str, Any]:
        return {
            "video": vp,
            "video_label": ctx["video_label"],
            "start_sec": round(cs, 2),
            "end_sec": round(ce, 2),
            "ground_truth": gt,
        }

    return collect_clip_features(
        video_entries(),
        models,
        device,
        num_frames=num_frames,
        frame_step=frame_step,
        human_threshold=human_threshold,
        weapon_threshold=weapon_threshold,
        label_fn=label_fn,
        metadata_fn=metadata_fn,
        strategy=strategy,
    )


def _collect_loss_curve(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> List[float]:
    if isinstance(model, GradientBoostingClassifier):
        return [
            float(log_loss(y, proba[:, 1]))
            for proba in model.staged_predict_proba(X)
        ]
    if isinstance(model, XGBClassifier):
        evals = getattr(model, "evals_result_", None) or {}
        return [
            float(v) for v in evals.get("validation_0", {}).get("logloss", [])
        ]
    return []


def _log_curve(name: str, curve: List[float], every: int) -> None:
    if every <= 0 or not curve:
        return
    n = len(curve)
    for i, ll in enumerate(curve, start=1):
        if i % every == 0 or i == n:
            logger.info(f"    {name} epoch {i}/{n}: train_loss={ll:.4f}")


def train_and_evaluate_stacking(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    log_loss_every: int = 10,
) -> Dict[str, Any]:
    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    scale = neg_count / pos_count if pos_count > 0 else 1.0

    meta_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=random_state,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
    }

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    results = {}

    for name, model in meta_models.items():
        logger.info(f"Training meta-model: {name} ({n_splits}-fold OOF)")
        try:
            oof_proba = cross_val_predict(
                model, X, y, cv=skf, method="predict_proba"
            )
            oof_preds = oof_proba.argmax(axis=1)
            metrics = compute_eval_metrics(y, oof_preds)
            oof_logloss = float(log_loss(y, oof_proba[:, 1]))
            if isinstance(model, XGBClassifier):
                model.fit(X, y, eval_set=[(X, y)], verbose=False)
            else:
                model.fit(X, y)
            loss_curve = _collect_loss_curve(model, X, y)
            _log_curve(name, loss_curve, log_loss_every)

            results[name] = {
                "metrics": metrics,
                "logloss": oof_logloss,
                "loss_curve": loss_curve,
                "predictions": oof_preds.tolist(),
                "model_instance": model,
            }
            logger.info(
                f"  {name}: F1={metrics['f1']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"Rec={metrics['recall']:.4f}  "
                f"LogLoss={oof_logloss:.4f}"
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
    for name, entry in results.items():
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
        k
        for k in results
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
    has_val: bool = False,
    raw_dim: int = 0,
    final_dim: int = 0,
) -> None:
    print("\n" + "=" * 82)
    print("  EMBEDDING META-MODEL STACKING RESULTS")
    if raw_dim:
        print(f"  (raw dim={raw_dim}, final dim={final_dim})")
    print("=" * 82)

    _print_section(results, "metrics", "Train (OOF)")

    if has_val:
        _print_section(results, "val_metrics", "Validation")

    print("=" * 82 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train embedding-based meta-learner (V-JEPA + YOLO-CLS)."
    )
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
        help="Path to validation CSV (same format).",
    )
    add_shared_model_args(p)
    add_strategy_arg(p)
    p.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="PCA components (0 = no PCA, use raw embeddings)",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--log-loss-every",
        type=int,
        default=10,
        help="Log per-iteration train logloss every N iters "
        "(GradientBoosting/XGBoost only; 0 = off)",
    )
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

    models = load_pipeline_models(
        args, device, strategy=args.strategy
    )

    # Collect training embeddings
    logger.info(f"Collecting per-clip embeddings (strategy={args.strategy.value}) …")
    t0 = time.time()
    X, y, metadata = collect_embeddings_from_csv(
        args.dataset_csv,
        models,
        device,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
        human_threshold=args.human_threshold,
        weapon_threshold=args.weapon_threshold,
        strategy=args.strategy,
    )
    collect_time = time.time() - t0
    logger.info(f"Embedding collection done in {collect_time:.1f}s")
    logger.info(f"Raw embedding shape: {X.shape}")

    # ── Preprocessing: StandardScaler + optional PCA ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = None
    if args.pca_dim > 0 and args.pca_dim < X.shape[1]:
        logger.info(f"Applying PCA: {X.shape[1]} → {args.pca_dim}")
        pca = PCA(n_components=args.pca_dim, random_state=42)
        X_train = pca.fit_transform(X_scaled)
        var_explained = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA variance explained: {var_explained:.4f}")
    else:
        X_train = X_scaled

    raw_dim = X.shape[1]
    final_dim = X_train.shape[1]
    logger.info(f"Final feature shape: {X_train.shape}")

    # ── Train meta-models ──
    logger.info("Training meta-models with OOF cross-validation …")
    t1 = time.time()
    results = train_and_evaluate_stacking(
        X_train,
        y,
        n_splits=args.n_folds,
        log_loss_every=args.log_loss_every,
    )
    train_time = time.time() - t1

    # ── Validation ──
    val_time = 0.0
    if has_val:
        logger.info("Collecting val-set embeddings …")
        tv = time.time()
        X_val, y_val, val_metadata = collect_embeddings_from_csv(
            args.val_csv,
            models,
            device,
            num_frames=args.num_frames,
            frame_step=args.frame_step,
            human_threshold=args.human_threshold,
            weapon_threshold=args.weapon_threshold,
            strategy=args.strategy,
        )
        X_val_scaled = scaler.transform(X_val)
        X_val_final = pca.transform(X_val_scaled) if pca else X_val_scaled

        logger.info("Evaluating meta-models on validation set …")
        results = evaluate_on_val(results, X_val_final, y_val)
        val_time = time.time() - tv

    print_comparison_table(
        results, has_val=has_val, raw_dim=raw_dim, final_dim=final_dim
    )

    # ── Save best model ──
    meta_only = {
        k: v
        for k, v in results.items()
        if "metrics" in v and "model_instance" in v
    }
    if meta_only:
        select_key = "val_metrics" if has_val else "metrics"
        candidates = {
            k: v
            for k, v in meta_only.items()
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
            "strategy": args.strategy.value,
            "scaler": scaler,
            "pca": pca,
            "pca_dim": args.pca_dim,
            "raw_dim": raw_dim,
            "final_dim": final_dim,
            "train_metrics": candidates[best_name]["metrics"],
        }
        if has_val and "val_metrics" in candidates[best_name]:
            save_payload["val_metrics"] = candidates[best_name]["val_metrics"]
        with open(args.save_model, "wb") as f:
            pickle.dump(save_payload, f)
        logger.info(
            f"Best meta-model ({best_name}) saved to: {args.save_model}"
        )

    # ── Save report ──
    serializable = {}
    for k, v in results.items():
        entry = {kk: vv for kk, vv in v.items() if kk != "model_instance"}
        serializable[k] = entry

    report: Dict[str, Any] = {
        "config": {
            "dataset_csv": os.path.abspath(args.dataset_csv),
            "val_csv": os.path.abspath(args.val_csv) if has_val else None,
            "strategy": args.strategy.value,
            "yolo_violence": args.yolo_violence,
            "encoder_weight": args.encoder_weight,
            "probe_weight": args.probe_weight,
            "pca_dim": args.pca_dim,
            "raw_embedding_dim": raw_dim,
            "final_dim": final_dim,
            "num_frames": args.num_frames,
            "frame_step": args.frame_step,
            "n_folds": args.n_folds,
            "device": device,
            "collect_time_sec": round(collect_time, 1),
            "train_time_sec": round(train_time, 1),
            "val_time_sec": round(val_time, 1),
        },
        "dataset_stats": {
            "train_clips": len(y),
            "train_positive": int(y.sum()),
            "train_negative": int(len(y) - y.sum()),
            "n_features": final_dim,
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
        write_clips_csv(args.output_csv, metadata)
        logger.info(f"Per-clip CSV saved to: {args.output_csv}")

    logger.info(
        f"Done — collection: {collect_time:.1f}s, "
        f"training: {train_time:.1f}s, val: {val_time:.1f}s"
    )


if __name__ == "__main__":
    main()
