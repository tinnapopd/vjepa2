"""Meta-model inference using V-JEPA + YOLO-CLS embeddings.

Tests a trained embedding-based meta-learner on a new dataset.

Usage:
    python meta_inference.py \\
        --test-dataset /tf/data/test-dataset \\
        --meta-model meta_model.pkl \\
        --cls-checkpoint trained-models/yolo26m-violence-cls/best.pt
"""

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
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

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


def load_labels(csv_path: str) -> List[Tuple[float, float]]:
    """Security Pitch's Custom Label Format

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of tuples, where each tuple contains the start and end time
        of a label segment
    """
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


def collect_embeddings(
    dataset_dir: str,
    models: PipelineModels,
    device: str,
    *,
    num_frames: int,
    frame_step: int,
    human_threshold: float = 0.3,
    weapon_threshold: float = 0.41,
    strategy: Optional[PipelineStrategy] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    def video_entries() -> Iterator[Tuple[str, Dict[str, Any]]]:
        for folder, is_pos_folder in [
            ("violent", True),
            ("non-violent", False),
        ]:
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
                yield (
                    vp,
                    {
                        "folder": folder,
                        "is_pos_folder": is_pos_folder,
                        "labels": load_labels(lpath),
                        "vname": vname,
                    },
                )

    def label_fn(ctx: Dict[str, Any], cs: float, ce: float) -> int:
        return (
            1
            if ctx["is_pos_folder"]
            and clip_overlaps_any_label(cs, ce, ctx["labels"])
            else 0
        )

    def metadata_fn(
        vp: str, ctx: Dict[str, Any], cs: float, ce: float, gt: int
    ) -> Dict[str, Any]:
        return {
            "video": ctx["vname"],
            "folder": ctx["folder"],
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


def print_results(
    meta_metrics: Dict[str, Any],
    model_name: str,
) -> None:
    print("\n" + "=" * 78)
    print(f"  EMBEDDING META-MODEL INFERENCE — {model_name}")
    print("=" * 78)

    m = meta_metrics
    print(
        f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}\n"
        f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}\n"
        f"  F1={m['f1']:.4f}  F2={m['f2']:.4f}\n"
        f"  Specificity={m['specificity']:.4f}  "
        f"Accuracy={m['accuracy']:.4f}"
    )

    print("=" * 78 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Test an embedding-based meta-learner on a new dataset."
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
        help="Path to saved meta_model.pkl from meta_training.py",
    )
    p.add_argument(
        "--cls-checkpoint",
        type=str,
        default="/tf/data/trained-models/yolo26m-violence-cls/best.pt",
        help="Path to trained YOLO-CLS violence classifier",
    )
    add_shared_model_args(p)
    add_strategy_arg(p)
    p.add_argument("--output", type=str, default="inference_report.json")
    p.add_argument("--output-csv", type=str, default="inference_clips.csv")
    args = p.parse_args()

    logger.info(f"Loading meta-model from: {args.meta_model}")
    with open(args.meta_model, "rb") as f:
        saved = pickle.load(f)

    meta_clf = saved["model"]
    model_name = saved["model_name"]
    scaler = saved["scaler"]
    pca = saved.get("pca")
    train_metrics = saved["train_metrics"]
    saved_strategy = saved.get("strategy", "combined")

    # Validate strategy consistency
    if args.strategy.value != saved_strategy:
        logger.warning(
            f"CLI strategy ({args.strategy.value}) differs from saved "
            f"model strategy ({saved_strategy}). Using saved: {saved_strategy}"
        )
        effective_strategy = PipelineStrategy(saved_strategy)
    else:
        effective_strategy = args.strategy

    logger.info(
        f"Loaded {model_name} (strategy={saved_strategy}) — "
        f"train F1={train_metrics['f1']}, "
        f"raw_dim={saved.get('raw_dim', '?')}, "
        f"final_dim={saved.get('final_dim', '?')}"
    )

    for sub in ("violent/videos", "non-violent/videos"):
        if not os.path.isdir(os.path.join(args.test_dataset, sub)):
            raise SystemExit(
                f"Error: not found: {os.path.join(args.test_dataset, sub)}"
            )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    models = load_pipeline_models(
        args, device, args.cls_checkpoint, strategy=effective_strategy
    )

    logger.info("Extracting embeddings from test dataset …")
    t0 = time.time()
    X, y, metadata = collect_embeddings(
        args.test_dataset,
        models,
        device,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
        human_threshold=args.human_threshold,
        weapon_threshold=args.weapon_threshold,
        strategy=effective_strategy,
    )
    elapsed = time.time() - t0

    # Apply same preprocessing as training
    X_scaled = scaler.transform(X)
    X_final = pca.transform(X_scaled) if pca else X_scaled

    meta_preds = meta_clf.predict(X_final)
    meta_metrics = compute_eval_metrics(y, meta_preds)

    print_results(meta_metrics, model_name)

    report: Dict[str, Any] = {
        "config": {
            "test_dataset": os.path.abspath(args.test_dataset),
            "meta_model": args.meta_model,
            "model_name": model_name,
            "strategy": effective_strategy.value,
            "raw_dim": saved.get("raw_dim"),
            "final_dim": saved.get("final_dim"),
            "pca_dim": saved.get("pca_dim"),
            "encoder_weight": args.encoder_weight,
            "probe_weight": args.probe_weight,
            "cls_checkpoint": args.cls_checkpoint,
            "num_frames": args.num_frames,
            "frame_step": args.frame_step,
            "device": device,
            "elapsed_sec": round(elapsed, 1),
        },
        "dataset_stats": {
            "test_clips": len(y),
            "test_positive": int(y.sum()),
            "test_negative": int(len(y) - y.sum()),
        },
        "train_metrics": train_metrics,
        "test_metrics": meta_metrics,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report saved to: {args.output}")

    if metadata:
        for i, clip in enumerate(metadata):
            clip["meta_pred"] = int(meta_preds[i])
        write_clips_csv(args.output_csv, metadata)
        logger.info(f"Per-clip CSV saved to: {args.output_csv}")

    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
