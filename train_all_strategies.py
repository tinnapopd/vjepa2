#!/usr/bin/env python3
"""Train and compare all meta-model strategies.

Loads the necessary deep learning models once, runs each selected strategy (using
cached features if available), trains the meta-classifiers, saves individual models,
and outputs a comparative performance summary.

Usage:
    python train_all_strategies.py \\
        --dataset-csv /tf/data/dataset-2classes/train.csv \\
        --val-csv /tf/data/dataset-2classes/val.csv \\
        --pca-dim 128
"""

import argparse
import json
import logging
import os
import pickle
import warnings
from typing import Any, Dict

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from meta_common import (  # type: ignore
    PipelineStrategy,
    add_shared_model_args,
    load_pipeline_models,
)
from meta_training import (  # type: ignore
    collect_embeddings_from_csv,
    evaluate_on_val,
    get_cache_path,
    train_and_evaluate_stacking,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig()
logger = logging.getLogger("train_all_strategies")
logger.setLevel(logging.INFO)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train and compare multiple embedding-based meta-learner strategies."
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
    p.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=[
            "all",
            "human_vjepa",
            "human_yolo_cls",
            "human_weapon_cls",
            "combined",
        ],
        help="Pipeline strategy to train, or 'all' to train all strategies sequentially",
    )
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
        default=0,
        help="Log per-iteration train logloss every N iters (0 = off)",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="cache_features",
        help="Directory to cache extracted features (empty string to disable)",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding features and ignore existing cache",
    )
    p.add_argument(
        "--output-report",
        type=str,
        default="all_strategies_report.json",
        help="Path to save comparative JSON report",
    )
    args = p.parse_args()

    if not os.path.isfile(args.dataset_csv):
        raise SystemExit(f"Error: CSV not found: {args.dataset_csv}")

    has_val = os.path.isfile(args.val_csv)
    if not has_val:
        logger.warning(
            f"Val CSV not found: {args.val_csv} — will select best models by OOF F1 only"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Map selected strategies
    if args.strategy == "all":
        strategies_to_run = [
            PipelineStrategy.HUMAN_VJEPA,
            PipelineStrategy.HUMAN_YOLO_CLS,
            PipelineStrategy.HUMAN_WEAPON_CLS,
            PipelineStrategy.COMBINED,
        ]
    else:
        strategies_to_run = [PipelineStrategy(args.strategy)]

    logger.info(f"Strategies to evaluate: {[s.value for s in strategies_to_run]}")

    # Load ALL pipeline models once by asking for COMBINED strategy models
    logger.info("Loading all required neural network models onto device once ...")
    models = load_pipeline_models(args, device, strategy=PipelineStrategy.COMBINED)

    comparison_results: Dict[str, Any] = {}
    master_table_rows = []

    for strat in strategies_to_run:
        logger.info("\n" + "=" * 60)
        logger.info(f" STARTING STRATEGY: {strat.value}")
        logger.info("=" * 60)

        # Set args.strategy to the current PipelineStrategy enum so that
        # get_cache_path (which calls args.strategy.value) works correctly.
        args.strategy = strat

        # ── 1. Collect training embeddings ──
        cache_path_train = None
        loaded_from_cache_train = False
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            cache_path_train = get_cache_path(args.dataset_csv, args, prefix="train")

        if (
            cache_path_train
            and os.path.exists(cache_path_train)
            and not args.force_rebuild
        ):
            logger.info(f"Loading training features from cache: {cache_path_train}")
            try:
                with open(cache_path_train, "rb") as f:
                    cached_data = pickle.load(f)
                X, y, metadata = (
                    cached_data["X"],
                    cached_data["y"],
                    cached_data["metadata"],
                )
                loaded_from_cache_train = True
            except Exception as e:
                logger.warning(
                    f"Failed to load cache from {cache_path_train}: {e}. "
                    "Extracting features raw."
                )

        if not loaded_from_cache_train:
            logger.info(f"Extracting features raw (strategy={strat.value}) …")
            X, y, metadata = collect_embeddings_from_csv(
                args.dataset_csv,
                models,
                device,
                num_frames=args.num_frames,
                frame_step=args.frame_step,
                human_threshold=args.human_threshold,
                weapon_threshold=args.weapon_threshold,
                strategy=strat,
            )
            if cache_path_train:
                logger.info(f"Saving training features to cache: {cache_path_train}")
                try:
                    with open(cache_path_train, "wb") as f:
                        pickle.dump({"X": X, "y": y, "metadata": metadata}, f)
                except Exception as e:
                    logger.warning(f"Failed to save cache to {cache_path_train}: {e}")

        raw_dim = X.shape[1]

        # ── 2. Preprocessing ──
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = None
        if args.pca_dim > 0 and args.pca_dim < X.shape[1]:
            logger.info(f"Applying PCA: {X.shape[1]} → {args.pca_dim}")
            pca = PCA(n_components=args.pca_dim, random_state=42)
            X_train = pca.fit_transform(X_scaled)
        else:
            X_train = X_scaled

        final_dim = X_train.shape[1]

        # ── 3. Train meta-models ──
        logger.info("Training meta-models with OOF cross-validation …")
        results = train_and_evaluate_stacking(
            X_train,
            y,
            n_splits=args.n_folds,
            log_loss_every=args.log_loss_every,
        )

        # ── 4. Validation ──
        if has_val:
            cache_path_val = None
            loaded_from_cache_val = False
            if args.cache_dir:
                cache_path_val = get_cache_path(args.val_csv, args, prefix="val")

            if (
                cache_path_val
                and os.path.exists(cache_path_val)
                and not args.force_rebuild
            ):
                logger.info(f"Loading validation features from cache: {cache_path_val}")
                try:
                    with open(cache_path_val, "rb") as f:
                        cached_data = pickle.load(f)
                    X_val, y_val, val_metadata = (
                        cached_data["X"],
                        cached_data["y"],
                        cached_data["metadata"],
                    )
                    loaded_from_cache_val = True
                except Exception as e:
                    logger.warning(
                        f"Failed to load cache from {cache_path_val}: {e}. "
                        "Extracting features raw."
                    )

            if not loaded_from_cache_val:
                logger.info("Collecting val-set embeddings …")
                X_val, y_val, val_metadata = collect_embeddings_from_csv(
                    args.val_csv,
                    models,
                    device,
                    num_frames=args.num_frames,
                    frame_step=args.frame_step,
                    human_threshold=args.human_threshold,
                    weapon_threshold=args.weapon_threshold,
                    strategy=strat,
                )
                if cache_path_val:
                    logger.info(
                        f"Saving validation features to cache: {cache_path_val}"
                    )
                    try:
                        with open(cache_path_val, "wb") as f:
                            pickle.dump(
                                {"X": X_val, "y": y_val, "metadata": val_metadata}, f
                            )
                    except Exception as e:
                        logger.warning(f"Failed to save cache to {cache_path_val}: {e}")

            X_val_scaled = scaler.transform(X_val)
            X_val_final = pca.transform(X_val_scaled) if pca else X_val_scaled

            logger.info("Evaluating meta-models on validation set …")
            results = evaluate_on_val(results, X_val_final, y_val)

        # ── 5. Save best model for this strategy ──
        meta_only = {
            k: v for k, v in results.items() if "metrics" in v and "model_instance" in v
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
            best_metrics = candidates[best_name]

            # Save strategy-specific meta-model
            model_save_path = f"meta_model_{strat.value}.pkl"
            save_payload = {
                "model": best_model,
                "model_name": best_name,
                "strategy": strat.value,
                "scaler": scaler,
                "pca": pca,
                "pca_dim": args.pca_dim,
                "raw_dim": raw_dim,
                "final_dim": final_dim,
                "train_metrics": best_metrics["metrics"],
            }
            if has_val and "val_metrics" in best_metrics:
                save_payload["val_metrics"] = best_metrics["val_metrics"]

            with open(model_save_path, "wb") as f:
                pickle.dump(save_payload, f)
            logger.info(
                f"Best meta-model ({best_name}) for strategy '{strat.value}' "
                f"saved to: {model_save_path}"
            )

            # Record stats for master table
            oof_f1 = best_metrics["metrics"].get("f1", 0.0)
            oof_prec = best_metrics["metrics"].get("precision", 0.0)
            oof_rec = best_metrics["metrics"].get("recall", 0.0)

            val_f1 = 0.0
            val_prec = 0.0
            val_rec = 0.0
            if has_val and "val_metrics" in best_metrics:
                val_f1 = best_metrics["val_metrics"].get("f1", 0.0)
                val_prec = best_metrics["val_metrics"].get("precision", 0.0)
                val_rec = best_metrics["val_metrics"].get("recall", 0.0)

            master_table_rows.append(
                {
                    "strategy": strat.value,
                    "best_model": best_name,
                    "oof_f1": oof_f1,
                    "oof_prec": oof_prec,
                    "oof_rec": oof_rec,
                    "val_f1": val_f1,
                    "val_prec": val_prec,
                    "val_rec": val_rec,
                }
            )

            # Save serializable results
            serializable_results = {}
            for k, v in results.items():
                serializable_results[k] = {
                    kk: vv for kk, vv in v.items() if kk != "model_instance"
                }

            comparison_results[strat.value] = {
                "best_model_name": best_name,
                "raw_dim": raw_dim,
                "final_dim": final_dim,
                "results": serializable_results,
            }

    # ── 6. Print Master Comparison Table ──
    print("\n" + "=" * 84)
    print(" " * 22 + "MASTER COMPARISON OF STRATEGIES")
    print("=" * 84)
    if has_val:
        header = (
            f"  {'Strategy':<20} {'Best Model':<20} "
            f"{'OOF F1':>8} {'OOF Prec':>8} {'Val F1':>8} {'Val Prec':>8}"
        )
    else:
        header = (
            f"  {'Strategy':<20} {'Best Model':<20} "
            f"{'OOF F1':>8} {'OOF Prec':>8} {'OOF Rec':>8}"
        )

    print(header)
    print("  " + "-" * 80)

    for row in master_table_rows:
        if has_val:
            line = (
                f"  {row['strategy']:<20} {row['best_model']:<20} "
                f"{row['oof_f1']:>8.3f} {row['oof_prec']:>8.3f} "
                f"{row['val_f1']:>8.3f} {row['val_prec']:>8.3f}"
            )
        else:
            line = (
                f"  {row['strategy']:<20} {row['best_model']:<20} "
                f"{row['oof_f1']:>8.3f} {row['oof_prec']:>8.3f} "
                f"{row['oof_rec']:>8.3f}"
            )
        print(line)

    print("=" * 84 + "\n")

    # Export to JSON
    report = {
        "config": {
            "dataset_csv": os.path.abspath(args.dataset_csv),
            "val_csv": os.path.abspath(args.val_csv) if has_val else None,
            "pca_dim": args.pca_dim,
            "n_folds": args.n_folds,
            "device": device,
        },
        "summary": master_table_rows,
        "details": comparison_results,
    }
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved comparative master report to: {args.output_report}")


if __name__ == "__main__":
    main()
