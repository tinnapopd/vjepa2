"""Multi-strategy evaluation: run all pipeline strategies side-by-side.

Evaluates each 2-phase strategy on a test dataset and produces a
comparison table with per-strategy confusion matrices and metrics.

Strategies evaluated:
  A) human_vjepa      — YOLO Human → V-JEPA temporal classification
  B) human_yolo_cls   — YOLO Human → YOLO-CLS frame-level classification
  C) human_weapon_cls — YOLO Human → YOLO Weapon Det → YOLO-CLS

Usage:
    python pipeline_strategy_eval.py \\
        --test-dataset /tf/data/test-dataset \\
        --yolo_violence trained-models/yolo26m-violence-cls/best.pt
"""

import argparse
import json
import logging
import os
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from meta_common import (  # type: ignore
    PipelineModels,
    PipelineStrategy,
    add_shared_model_args,
    compute_eval_metrics,
    extract_strategy_features,
    has_human_in_clip,
    iter_clips_from_video,
    load_pipeline_models,
    write_clips_csv,
)
from meta_inference import (  # type: ignore
    clip_overlaps_any_label,
    find_videos,
    load_labels,
)
from meta_training import parse_dataset_csv  # type: ignore

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def collect_all_strategies(
    dataset_dir: Optional[str],
    csv_path: Optional[str],
    models: PipelineModels,
    device: str,
    *,
    num_frames: int,
    frame_step: int,
    human_threshold: float = 0.3,
    weapon_threshold: float = 0.41,
    strategies: Optional[List[PipelineStrategy]] = None,
) -> Dict[str, Any]:
    """Walk the dataset (either directory or CSV list), extract features for ALL strategies.

    Returns a dict with:
      - per_strategy: {strategy_name: {"X": ndarray, "y": ndarray}}
      - metadata: list of per-clip metadata dicts
      - skipped_no_human: int
    """
    if strategies is None:
        strategies = list(PipelineStrategy)

    # Accumulators per strategy
    features: Dict[str, List[np.ndarray]] = {s.value: [] for s in strategies}
    all_labels: List[int] = []
    metadata: List[Dict[str, Any]] = []
    skipped_no_human = 0

    if csv_path:
        entries = parse_dataset_csv(csv_path)
        logger.info(f"Loaded {len(entries)} videos from {csv_path}")

        for vi, (vp, video_label) in enumerate(entries):
            vname = os.path.basename(vp)
            logger.info(f"  [{vi + 1}/{len(entries)}] {vname}")

            for bgr, rgb, cs_sec, ce_sec in iter_clips_from_video(
                vp, num_frames=num_frames, frame_step=frame_step
            ):
                # Phase 1: Human gate
                has_human, _ = has_human_in_clip(
                    bgr, models.human_model, human_threshold
                )
                if not has_human:
                    skipped_no_human += 1
                    continue

                # Ground truth: 1 if video_label matches model's positive_idx, else 0
                gt = 1 if video_label == models.positive_idx else 0
                all_labels.append(gt)

                folder_name = "weaponized" if video_label == models.positive_idx else "background"
                metadata.append(
                    {
                        "video": vp,
                        "folder": folder_name,
                        "start_sec": round(cs_sec, 2),
                        "end_sec": round(ce_sec, 2),
                        "ground_truth": gt,
                    }
                )

                # Phase 2: Extract features for each strategy
                for strat in strategies:
                    emb = extract_strategy_features(
                        strat,
                        rgb,
                        bgr,
                        models.encoder,
                        models.cls_model,
                        device,
                        weapon_model=models.weapon_model,
                        weapon_threshold=weapon_threshold,
                    )
                    features[strat.value].append(emb)

    elif dataset_dir:
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
                lpath = os.path.join(
                    ldir, f"{os.path.splitext(os.path.basename(vp))[0]}.csv"
                )
                labels = load_labels(lpath)

                for bgr, rgb, cs_sec, ce_sec in iter_clips_from_video(
                    vp, num_frames=num_frames, frame_step=frame_step
                ):
                    # Phase 1: Human gate
                    has_human, _ = has_human_in_clip(
                        bgr, models.human_model, human_threshold
                    )
                    if not has_human:
                        skipped_no_human += 1
                        continue

                    # Ground truth
                    if is_pos_folder and clip_overlaps_any_label(
                        cs_sec, ce_sec, labels
                    ):
                        gt = 1
                    else:
                        gt = 0
                    all_labels.append(gt)

                    metadata.append(
                        {
                            "video": vname,
                            "folder": folder,
                            "start_sec": round(cs_sec, 2),
                            "end_sec": round(ce_sec, 2),
                            "ground_truth": gt,
                        }
                    )

                    # Phase 2: Extract features for each strategy
                    for strat in strategies:
                        emb = extract_strategy_features(
                            strat,
                            rgb,
                            bgr,
                            models.encoder,
                            models.cls_model,
                            device,
                            weapon_model=models.weapon_model,
                            weapon_threshold=weapon_threshold,
                        )
                        features[strat.value].append(emb)
    else:
        raise ValueError("Either dataset_dir or csv_path must be provided")

    if skipped_no_human:
        logger.info(f"Skipped {skipped_no_human} clips (no human detected)")

    y = np.array(all_labels, dtype=np.int32)
    per_strategy: Dict[str, Dict[str, np.ndarray]] = {}
    for strat in strategies:
        if features[strat.value]:
            X = np.stack(features[strat.value], axis=0)
            per_strategy[strat.value] = {"X": X, "y": y}
            logger.info(
                f"  {strat.value}: {X.shape[0]} clips, dim={X.shape[1]}"
            )

    return {
        "per_strategy": per_strategy,
        "metadata": metadata,
        "skipped_no_human": skipped_no_human,
    }


def train_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Quick LogisticRegression train→predict for strategy comparison."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000, random_state=42, class_weight="balanced"
    )
    clf.fit(X_tr, y_train)
    return clf.predict(X_te)


def print_comparison(
    results: Dict[str, Dict[str, Any]],
    total_clips: int,
    skipped: int,
) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 82)
    print("  MULTI-STRATEGY EVALUATION RESULTS")
    print(
        f"  Clips evaluated: {total_clips}  |  Skipped (no human): {skipped}"
    )
    print("=" * 82)

    header = (
        f"  {'Strategy':<22}"
        f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'F2':>6} {'Spec':>6}"
    )
    print(header)
    print("  " + "-" * 78)

    sorted_strats = sorted(
        results.keys(),
        key=lambda k: results[k].get("f1", 0),
        reverse=True,
    )
    best = sorted_strats[0] if sorted_strats else ""

    for sname in sorted_strats:
        m = results[sname]
        marker = " ★" if sname == best else ""
        print(
            f"  {sname:<22}"
            f"{m['tp']:>5} {m['fp']:>5} {m['tn']:>5} {m['fn']:>5}  "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['f2']:>6.3f} "
            f"{m['specificity']:>6.3f}{marker}"
        )

    print("=" * 82 + "\n")



def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run all pipeline strategies side-by-side and compare results."
        )
    )
    p.add_argument(
        "--test-dataset",
        type=str,
        default=None,
        help="Path to test dataset (violent/ + non-violent/ subdirs)",
    )
    p.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Path to test CSV mapping video files to class labels",
    )
    add_shared_model_args(p)
    p.add_argument(
        "--strategy",
        type=str,
        default="all",
        help="Specific strategy to evaluate (e.g., human_vjepa). Default is 'all'.",
    )
    p.add_argument("--output", type=str, default="strategy_eval_report.json")
    p.add_argument("--output-csv", type=str, default="strategy_eval_clips.csv")
    args = p.parse_args()

    if not args.test_dataset and not args.test_csv:
        raise SystemExit("Error: Must specify either --test-dataset or --test-csv")

    # Validate dataset
    if args.test_dataset:
        for sub in ("violent/videos", "non-violent/videos"):
            if not os.path.isdir(os.path.join(args.test_dataset, sub)):
                raise SystemExit(
                    f"Error: not found: {os.path.join(args.test_dataset, sub)}"
                )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    eval_strategies = list(PipelineStrategy)
    if args.strategy != "all":
        try:
            eval_strategies = [PipelineStrategy(args.strategy)]
        except ValueError:
            raise SystemExit(f"Invalid strategy: {args.strategy}")

    # Load required models (strategy=None loads everything)
    models = load_pipeline_models(
        args,
        device,
        strategy=eval_strategies[0] if len(eval_strategies) == 1 else None,
    )

    # Extract features for all strategies in one pass
    logger.info("Extracting features for all strategies …")
    t0 = time.time()
    data = collect_all_strategies(
        args.test_dataset,
        args.test_csv,
        models,
        device,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
        human_threshold=args.human_threshold,
        weapon_threshold=args.weapon_threshold,
        strategies=eval_strategies,
    )
    elapsed = time.time() - t0
    logger.info(f"Feature extraction done in {elapsed:.1f}s")

    per_strategy = data["per_strategy"]
    metadata = data["metadata"]
    skipped = data["skipped_no_human"]

    # For each strategy, do a simple train-on-self evaluation
    # (in practice you'd use a separate train set, but this gives
    #  a quick apples-to-apples comparison of feature quality)
    all_metrics: Dict[str, Dict[str, Any]] = {}

    for strat_name, strat_data in per_strategy.items():
        X = strat_data["X"]
        y = strat_data["y"]
        logger.info(
            f"Evaluating {strat_name}: {X.shape[0]} clips, dim={X.shape[1]}"
        )

        # Use cross-validated predictions for fair comparison
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced"
        )
        try:
            preds = cross_val_predict(clf, X_scaled, y, cv=5)
            metrics = compute_eval_metrics(y, preds)
            all_metrics[strat_name] = metrics
            logger.info(
                f"  {strat_name}: F1={metrics['f1']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"Rec={metrics['recall']:.4f}"
            )
        except Exception as e:
            logger.error(f"  {strat_name} failed: {e}")
            all_metrics[strat_name] = {"error": str(e)}

    total_clips = len(metadata)
    print_comparison(all_metrics, total_clips, skipped)

    # Save report
    report: Dict[str, Any] = {
        "config": {
            "test_dataset": os.path.abspath(args.test_dataset) if args.test_dataset else None,
            "test_csv": os.path.abspath(args.test_csv) if args.test_csv else None,
            "yolo_violence": args.yolo_violence,
            "encoder_weight": args.encoder_weight,
            "num_frames": args.num_frames,
            "frame_step": args.frame_step,
            "device": device,
            "elapsed_sec": round(elapsed, 1),
        },
        "dataset_stats": {
            "total_clips": total_clips,
            "positive_clips": int(
                sum(1 for m in metadata if m["ground_truth"] == 1)
            ),
            "negative_clips": int(
                sum(1 for m in metadata if m["ground_truth"] == 0)
            ),
            "skipped_no_human": skipped,
        },
        "strategy_dims": {
            sname: int(sdata["X"].shape[1])
            for sname, sdata in per_strategy.items()
        },
        "results": all_metrics,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report saved to: {args.output}")

    if metadata:
        write_clips_csv(args.output_csv, metadata)
        logger.info(f"Per-clip CSV saved to: {args.output_csv}")

    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
