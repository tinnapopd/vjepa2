"""Frame-rate influence analysis: is the val→test accuracy drop driven by fps?

Clips are cut with a fixed `frame_step` measured in FRAMES, not seconds
(see meta_common.iter_clips_from_video). A clip therefore spans
`num_frames * frame_step / fps` seconds — so two videos at different native
frame rates yield different real-world temporal windows for the SAME
frame_step. If the test videos have a systematically different fps than the
val videos, the model sees a different temporal scale at inference time.

This script investigates that with two analyses, using your trained meta-model:

  A) Native fps distribution — val vs test. If these differ, fixed frame_step
     means different effective temporal coverage.
  B) frame_step sweep — re-extract features at several frame_step values and
     evaluate the meta-model on val and test for each. If the test metric
     responds strongly to frame_step (and the val/test gap shrinks at some
     value), frame rate / temporal sampling is implicated in the drop.

Validation features are reused from cache_features/ per (strategy, frame_step)
when available.

Usage:
    python frame_rate_analysis.py \\
        --val-csv /tf/data/dataset-2classes/val.csv \\
        --test-dataset /tf/data/test-dataset \\
        --meta-model meta_model_human_weapon_cls.pkl \\
        --frame-steps 2,4,8,16 \\
        --yolo_violence trained-models/yolo26m-violence-cls/best.pt
"""

import argparse
import json
import logging
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from meta_common import (  # type: ignore
    DEFAULT_STRATEGY,
    PipelineStrategy,
    add_shared_model_args,
    aggregate_clips_to_videos,
    compute_eval_metrics,
    load_pipeline_models,
)
from meta_inference import collect_embeddings, find_videos  # type: ignore
from meta_training import (  # type: ignore
    collect_embeddings_from_csv,
    get_cache_path,
    parse_dataset_csv,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig()
logger = logging.getLogger("frame_rate_analysis")
logger.setLevel(logging.INFO)


def get_video_fps(path: str) -> Optional[float]:
    """Return native average fps for a video, or None if unreadable."""
    try:
        import decord  # type: ignore

        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        fps = float(vr.get_avg_fps())
        return fps if fps > 0 else None
    except Exception:
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps if fps > 0 else None
    except Exception:
        return None


def fps_summary(paths: List[str]) -> Dict[str, Any]:
    fps_vals = [f for f in (get_video_fps(p) for p in paths) if f]
    if not fps_vals:
        return {"n": 0, "fps": []}
    arr = np.array(fps_vals, dtype=np.float32)
    return {
        "n": int(len(arr)),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "std": round(float(arr.std()), 2),
        "fps": [round(float(x), 2) for x in arr],
    }


def collect_test_video_paths(test_dataset: str) -> List[str]:
    paths: List[str] = []
    for folder in ("violent", "non-violent"):
        vdir = os.path.join(test_dataset, folder, "videos")
        if os.path.isdir(vdir):
            paths.extend(find_videos(vdir))
    return paths


def _val_features(
    args: argparse.Namespace, models: Any, device: str, frame_step: int
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Val (X, y, metadata) at a given frame_step, reusing cache when present."""
    cache_path = None
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = get_cache_path(args.val_csv, args, prefix="val")
    if cache_path and os.path.exists(cache_path) and not args.force_rebuild:
        logger.info(f"  [val fs={frame_step}] cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached["X"], cached["y"], cached.get("metadata", [])
        except Exception as e:
            logger.warning(f"  cache load failed ({e}); extracting raw.")
    X, y, metadata = collect_embeddings_from_csv(
        args.val_csv, models, device,
        num_frames=args.num_frames, frame_step=frame_step,
        human_threshold=args.human_threshold,
        weapon_threshold=args.weapon_threshold,
        strategy=args.strategy,
    )
    if cache_path:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"X": X, "y": y, "metadata": metadata}, f)
        except Exception as e:
            logger.warning(f"  failed to save val cache: {e}")
    return X, y, metadata


def _test_features(
    args: argparse.Namespace, models: Any, device: str, frame_step: int
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Test (X, y, metadata) at a given frame_step, caching the result."""
    cache_path = None
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = get_cache_path(
            os.path.join(args.test_dataset, "test-dataset"),
            args, prefix="test",
        )
    if cache_path and os.path.exists(cache_path) and not args.force_rebuild:
        logger.info(f"  [test fs={frame_step}] cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached["X"], cached["y"], cached.get("metadata", [])
        except Exception as e:
            logger.warning(f"  cache load failed ({e}); extracting raw.")
    X, y, metadata = collect_embeddings(
        args.test_dataset, models, device,
        num_frames=args.num_frames, frame_step=frame_step,
        human_threshold=args.human_threshold,
        weapon_threshold=args.weapon_threshold,
        strategy=args.strategy,
    )
    if cache_path:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"X": X, "y": y, "metadata": metadata}, f)
        except Exception as e:
            logger.warning(f"  failed to save test cache: {e}")
    return X, y, metadata


def _maybe_aggregate(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict[str, Any]],
    eval_level: str,
    video_pool: str,
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool clips → videos when the model expects video-level features.

    Raises a clear error if a stale cache lacks the per-clip metadata needed
    to group by video (fix: re-run with --force-rebuild).
    """
    if eval_level != "video":
        return X, y
    if not metadata:
        raise SystemExit(
            f"Model was trained with eval_level=video but the cached {name} "
            "features have no per-clip metadata to group by video. "
            "Re-run with --force-rebuild to rebuild the feature cache."
        )
    Xv, yv, _ = aggregate_clips_to_videos(X, y, metadata, pool=video_pool)
    return Xv, yv


def evaluate(
    clf: Any, scaler: Any, pca: Any, X: np.ndarray, y: np.ndarray
) -> Dict[str, Any]:
    X_s = scaler.transform(X)
    X_f = pca.transform(X_s) if pca else X_s
    preds = clf.predict(X_f)
    return compute_eval_metrics(y, preds)


def make_plot(
    rows: List[Dict[str, Any]],
    val_fps: Dict[str, Any],
    test_fps: Dict[str, Any],
    strategy: str,
    out_path: str,
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable ({e}); skipping plot.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: native fps distribution, val vs test ──
    if val_fps.get("fps") or test_fps.get("fps"):
        lo = min(
            [min(val_fps.get("fps", [30])), min(test_fps.get("fps", [30]))]
        )
        hi = max(
            [max(val_fps.get("fps", [30])), max(test_fps.get("fps", [30]))]
        )
        bins = np.linspace(lo - 1, hi + 1, 25)
        if val_fps.get("fps"):
            ax1.hist(
                val_fps["fps"], bins=bins, alpha=0.5, color="tab:blue",
                density=True, label=f"val (median {val_fps.get('median')})",
            )
        if test_fps.get("fps"):
            ax1.hist(
                test_fps["fps"], bins=bins, alpha=0.5, color="tab:red",
                density=True, label=f"test (median {test_fps.get('median')})",
            )
    ax1.set_xlabel("native fps")
    ax1.set_ylabel("density")
    ax1.set_title("Native frame-rate distribution: val vs test")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Right: F1 vs frame_step for val and test ──
    fsteps = [r["frame_step"] for r in rows]
    ax2.plot(
        fsteps, [r["val"]["f1"] for r in rows],
        "o-", color="tab:blue", label="val F1",
    )
    ax2.plot(
        fsteps, [r["test"]["f1"] for r in rows],
        "s-", color="tab:red", label="test F1",
    )
    ax2.plot(
        fsteps, [r["val"]["accuracy"] for r in rows],
        "o--", color="tab:blue", alpha=0.5, label="val acc",
    )
    ax2.plot(
        fsteps, [r["test"]["accuracy"] for r in rows],
        "s--", color="tab:red", alpha=0.5, label="test acc",
    )
    ax2.set_xlabel("frame_step (frames between sampled frames)")
    ax2.set_ylabel("score")
    ax2.set_title(f"Metric vs frame_step — strategy={strategy}")
    ax2.set_xticks(fsteps)
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def print_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("  FRAME-STEP SWEEP — meta-model on val vs test")
    print("=" * 78)
    print(
        f"  {'fstep':>5} | {'val F1':>7} {'val acc':>7} | "
        f"{'test F1':>7} {'test acc':>8} | {'F1 gap':>7}"
    )
    print("  " + "-" * 60)
    for r in rows:
        gap = r["val"]["f1"] - r["test"]["f1"]
        print(
            f"  {r['frame_step']:>5} | "
            f"{r['val']['f1']:>7.3f} {r['val']['accuracy']:>7.3f} | "
            f"{r['test']['f1']:>7.3f} {r['test']['accuracy']:>8.3f} | "
            f"{gap:>7.3f}"
        )
    print("=" * 78 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze frame-rate / frame_step influence on val→test drop."
    )
    p.add_argument(
        "--val-csv",
        type=str,
        default="/tf/data/dataset-2classes/val.csv",
        help="Validation CSV: '<video_path> <label>' per line.",
    )
    p.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="Test dataset dir (violent/ + non-violent/ subdirs).",
    )
    p.add_argument(
        "--meta-model",
        type=str,
        default="meta_model.pkl",
        help="Trained meta-model pickle (carries scaler/pca/strategy).",
    )
    p.add_argument(
        "--frame-steps",
        type=str,
        default="2,4,8,16",
        help="Comma-separated frame_step values to sweep.",
    )
    add_shared_model_args(p)
    p.add_argument(
        "--cache-dir",
        type=str,
        default="cache_features",
        help="Feature cache dir (shared with train_all_strategies.py).",
    )
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Only report the native fps distribution (analysis A), skip B.",
    )
    p.add_argument(
        "--output-plot", type=str, default="frame_rate_analysis.png"
    )
    p.add_argument("--output", type=str, default="frame_rate_analysis.json")
    args = p.parse_args()

    if not os.path.isfile(args.val_csv):
        raise SystemExit(f"Error: val CSV not found: {args.val_csv}")
    for sub in ("violent/videos", "non-violent/videos"):
        if not os.path.isdir(os.path.join(args.test_dataset, sub)):
            raise SystemExit(
                f"Error: not found: {os.path.join(args.test_dataset, sub)}"
            )

    # ── Analysis A: native fps distribution (cheap, no model needed) ──
    logger.info("Collecting native fps distribution (val vs test) …")
    val_paths = [vp for vp, _ in parse_dataset_csv(args.val_csv)]
    test_paths = collect_test_video_paths(args.test_dataset)
    val_fps = fps_summary(val_paths)
    test_fps = fps_summary(test_paths)
    logger.info(
        f"  val fps:  median={val_fps.get('median')} "
        f"range=[{val_fps.get('min')}, {val_fps.get('max')}] n={val_fps['n']}"
    )
    logger.info(
        f"  test fps: median={test_fps.get('median')} "
        f"range=[{test_fps.get('min')}, {test_fps.get('max')}] n={test_fps['n']}"
    )

    rows: List[Dict[str, Any]] = []
    strategy_label = "n/a"
    eval_level = "clip"
    video_pool = "mean"

    if not args.skip_sweep:
        # Load the trained meta-model (scaler/pca/clf/strategy).
        logger.info(f"Loading meta-model: {args.meta_model}")
        with open(args.meta_model, "rb") as f:
            saved = pickle.load(f)
        clf = saved["model"]
        scaler = saved["scaler"]
        pca = saved.get("pca")
        strategy = PipelineStrategy(
            saved.get("strategy", DEFAULT_STRATEGY.value)
        )
        args.strategy = strategy  # needed by get_cache_path / extraction
        strategy_label = strategy.value
        eval_level = saved.get("eval_level", "clip")
        video_pool = saved.get("video_pool", "mean")
        logger.info(
            f"  model={saved.get('model_name')} strategy={strategy.value} "
            f"eval_level={eval_level} video_pool={video_pool}"
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {device}")
        models = load_pipeline_models(args, device, strategy=strategy)

        frame_steps = [
            int(s) for s in args.frame_steps.split(",") if s.strip()
        ]
        logger.info(f"Sweeping frame_step values: {frame_steps}")

        for fs in frame_steps:
            logger.info(f"── frame_step={fs} ──")
            args.frame_step = fs  # drives get_cache_path key + extraction
            X_val, y_val, md_val = _val_features(args, models, device, fs)
            X_test, y_test, md_test = _test_features(args, models, device, fs)
            X_val, y_val = _maybe_aggregate(
                X_val, y_val, md_val, eval_level, video_pool, "val"
            )
            X_test, y_test = _maybe_aggregate(
                X_test, y_test, md_test, eval_level, video_pool, "test"
            )
            val_m = evaluate(clf, scaler, pca, X_val, y_val)
            test_m = evaluate(clf, scaler, pca, X_test, y_test)
            logger.info(
                f"  fs={fs}: val F1={val_m['f1']:.3f} "
                f"test F1={test_m['f1']:.3f} "
                f"(gap={val_m['f1'] - test_m['f1']:.3f})"
            )
            rows.append(
                {"frame_step": fs, "val": val_m, "test": test_m}
            )

        print_table(rows)

    plot_path = make_plot(
        rows, val_fps, test_fps, strategy_label, args.output_plot,
    ) if (rows or val_fps["n"] or test_fps["n"]) else None
    if plot_path:
        logger.info(f"Plot saved to: {plot_path}")

    report: Dict[str, Any] = {
        "config": {
            "val_csv": os.path.abspath(args.val_csv),
            "test_dataset": os.path.abspath(args.test_dataset),
            "meta_model": args.meta_model,
            "eval_level": eval_level,
            "video_pool": video_pool,
            "num_frames": args.num_frames,
            "frame_steps": args.frame_steps if not args.skip_sweep else None,
        },
        "native_fps": {
            "val": {k: v for k, v in val_fps.items() if k != "fps"},
            "test": {k: v for k, v in test_fps.items() if k != "fps"},
        },
        "frame_step_sweep": rows,
        "plot": plot_path,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report saved to: {args.output}")


if __name__ == "__main__":
    main()
