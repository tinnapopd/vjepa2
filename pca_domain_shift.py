"""PCA domain-shift analysis: validation set vs. test set.

Extracts embeddings (for a chosen pipeline strategy) from BOTH the
validation CSV and the test dataset directory, projects them into a shared
PCA(2) space, and renders a scatter plot so the domain gap between the two
distributions can be inspected visually. Also reports a quantitative
"domain-classifier AUC": how separable val and test are in feature space
(0.5 = indistinguishable / no shift, 1.0 = completely separable / large shift).

Validation features are loaded from the same cache_features/ directory used by
train_all_strategies.py when available, so they are not re-extracted.

Usage:
    python pca_domain_shift.py \\
        --val-csv /tf/data/dataset-2classes/val.csv \\
        --test-dataset /tf/data/test-dataset \\
        --strategy human_weapon_cls \\
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from meta_common import (  # type: ignore
    PipelineStrategy,
    add_shared_model_args,
    add_strategy_arg,
    add_video_level_args,
    aggregate_clips_to_videos,
    load_pipeline_models,
)
from meta_inference import collect_embeddings  # type: ignore
from meta_training import (  # type: ignore
    collect_embeddings_from_csv,
    get_cache_path,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig()
logger = logging.getLogger("pca_domain_shift")
logger.setLevel(logging.INFO)


def _load_or_extract_val(
    args: argparse.Namespace,
    models: Any,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Return (X, y, metadata) for the val CSV, reusing cache_features/."""
    cache_path = None
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = get_cache_path(args.val_csv, args, prefix="val")

    if cache_path and os.path.exists(cache_path) and not args.force_rebuild:
        logger.info(f"Loading val features from cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached["X"], cached["y"], cached.get("metadata", [])
        except Exception as e:
            logger.warning(f"Cache load failed ({e}); extracting raw.")

    logger.info("Extracting val embeddings …")
    X, y, metadata = collect_embeddings_from_csv(
        args.val_csv,
        models,
        device,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
        human_threshold=args.human_threshold,
        weapon_threshold=args.weapon_threshold,
        strategy=args.strategy,
    )
    if cache_path:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"X": X, "y": y, "metadata": metadata}, f)
        except Exception as e:
            logger.warning(f"Failed to save val cache: {e}")
    return X, y, metadata


def _load_or_extract_test(
    args: argparse.Namespace,
    models: Any,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Return (X, y, metadata) for the test dataset dir, caching the result."""
    cache_path = None
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        # Reuse get_cache_path's naming by treating the dir as a pseudo-CSV key.
        cache_path = get_cache_path(
            os.path.join(args.test_dataset, "test-dataset"),
            args,
            prefix="test",
        )

    if cache_path and os.path.exists(cache_path) and not args.force_rebuild:
        logger.info(f"Loading test features from cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached["X"], cached["y"], cached.get("metadata", [])
        except Exception as e:
            logger.warning(f"Cache load failed ({e}); extracting raw.")

    logger.info("Extracting test embeddings …")
    X, y, metadata = collect_embeddings(
        args.test_dataset,
        models,
        device,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
        human_threshold=args.human_threshold,
        weapon_threshold=args.weapon_threshold,
        strategy=args.strategy,
    )
    if cache_path:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"X": X, "y": y, "metadata": metadata}, f)
        except Exception as e:
            logger.warning(f"Failed to save test cache: {e}")
    return X, y, metadata


def domain_classifier_auc(
    X_val: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42,
) -> float:
    """Cross-validated AUC of a classifier trained to tell val from test.

    ~0.5 → distributions overlap (little shift); →1.0 → strongly separable
    (large domain shift).
    """
    X = np.vstack([X_val, X_test])
    domain = np.concatenate(
        [np.zeros(len(X_val)), np.ones(len(X_test))]
    ).astype(np.int32)
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=random_state
    )
    try:
        proba = cross_val_predict(
            clf, X, domain, cv=5, method="predict_proba"
        )[:, 1]
        return float(roc_auc_score(domain, proba))
    except Exception as e:
        logger.warning(f"Domain-classifier AUC failed: {e}")
        return float("nan")


def make_plot(
    pcs_val: np.ndarray,
    y_val: np.ndarray,
    pcs_test: np.ndarray,
    y_test: np.ndarray,
    var_ratio: np.ndarray,
    domain_auc: float,
    strategy: str,
    out_path: str,
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(
            f"matplotlib unavailable ({e}); skipping plot, saving data only."
        )
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: PC1 vs PC2 scatter, colored by domain, marker by class ──
    specs = [
        (pcs_val, y_val, "val", "tab:blue"),
        (pcs_test, y_test, "test", "tab:red"),
    ]
    for pcs, y, name, color in specs:
        for cls, marker in [(0, "o"), (1, "x")]:
            mask = y == cls
            if not mask.any():
                continue
            ax1.scatter(
                pcs[mask, 0],
                pcs[mask, 1],
                s=14,
                c=color,
                marker=marker,
                alpha=0.4,
                label=f"{name} ({'violent' if cls else 'non-violent'})",
            )
    ax1.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}% var)")
    ax1.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}% var)")
    ax1.set_title(
        f"PCA feature space — strategy={strategy}\n"
        f"domain-classifier AUC={domain_auc:.3f} "
        f"(0.5=no shift, 1.0=large shift)"
    )
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Right: PC1 marginal distribution, val vs test ──
    bins = np.linspace(
        min(pcs_val[:, 0].min(), pcs_test[:, 0].min()),
        max(pcs_val[:, 0].max(), pcs_test[:, 0].max()),
        50,
    )
    ax2.hist(
        pcs_val[:, 0], bins=bins, alpha=0.5, color="tab:blue",
        density=True, label="val",
    )
    ax2.hist(
        pcs_test[:, 0], bins=bins, alpha=0.5, color="tab:red",
        density=True, label="test",
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("density")
    ax2.set_title("PC1 marginal: val vs test (shift along dominant axis)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="PCA domain-shift analysis between val and test sets."
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
    add_shared_model_args(p)
    add_strategy_arg(p)
    add_video_level_args(p)
    p.add_argument(
        "--cache-dir",
        type=str,
        default="cache_features",
        help="Feature cache dir (shared with train_all_strategies.py).",
    )
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument(
        "--output-plot", type=str, default="pca_domain_shift.png"
    )
    p.add_argument("--output", type=str, default="pca_domain_shift.json")
    args = p.parse_args()

    if not os.path.isfile(args.val_csv):
        raise SystemExit(f"Error: val CSV not found: {args.val_csv}")
    for sub in ("violent/videos", "non-violent/videos"):
        if not os.path.isdir(os.path.join(args.test_dataset, sub)):
            raise SystemExit(
                f"Error: not found: {os.path.join(args.test_dataset, sub)}"
            )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | strategy: {args.strategy.value}")

    models = load_pipeline_models(args, device, strategy=args.strategy)

    X_val, y_val, md_val = _load_or_extract_val(args, models, device)
    X_test, y_test, md_test = _load_or_extract_test(args, models, device)

    # Pool clips → one point per video so each scatter point is a video,
    # matching the video-level training/eval convention.
    if args.eval_level == "video":
        for name, md in (("val", md_val), ("test", md_test)):
            if not md:
                raise SystemExit(
                    f"--eval-level=video needs per-clip metadata for {name}, "
                    "but the cache has none. Re-run with --force-rebuild."
                )
        X_val, y_val, _ = aggregate_clips_to_videos(
            X_val, y_val, md_val, pool=args.video_pool
        )
        X_test, y_test, _ = aggregate_clips_to_videos(
            X_test, y_test, md_test, pool=args.video_pool
        )

    unit = "videos" if args.eval_level == "video" else "clips"
    logger.info(
        f"val: {X_val.shape[0]} {unit} | test: {X_test.shape[0]} {unit} | "
        f"dim={X_val.shape[1]}"
    )

    # Standardize on the COMBINED set so both share one frame of reference,
    # then project both into a shared PCA(2) space.
    scaler = StandardScaler()
    X_all = scaler.fit_transform(np.vstack([X_val, X_test]))
    X_val_s = X_all[: len(X_val)]
    X_test_s = X_all[len(X_val):]

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_all)
    pcs_val = pca.transform(X_val_s)
    pcs_test = pca.transform(X_test_s)
    var_ratio = pca.explained_variance_ratio_

    domain_auc = domain_classifier_auc(X_val_s, X_test_s)
    logger.info(
        f"Domain-classifier AUC = {domain_auc:.4f} "
        f"(0.5=no shift, 1.0=large shift)"
    )
    logger.info(
        f"PC1/PC2 variance explained: "
        f"{var_ratio[0]:.3f} / {var_ratio[1]:.3f}"
    )

    plot_path = make_plot(
        pcs_val, y_val, pcs_test, y_test,
        var_ratio, domain_auc, args.strategy.value, args.output_plot,
    )
    if plot_path:
        logger.info(f"Plot saved to: {plot_path}")

    # Per-PC mean gap as an extra numeric summary of the shift.
    pc_mean_gap = [
        float(abs(pcs_val[:, i].mean() - pcs_test[:, i].mean()))
        for i in range(2)
    ]

    report: Dict[str, Any] = {
        "config": {
            "val_csv": os.path.abspath(args.val_csv),
            "test_dataset": os.path.abspath(args.test_dataset),
            "strategy": args.strategy.value,
            "eval_level": args.eval_level,
            "video_pool": args.video_pool,
            "num_frames": args.num_frames,
            "frame_step": args.frame_step,
            "raw_dim": int(X_val.shape[1]),
        },
        "stats": {
            "level": args.eval_level,
            "val_samples": int(len(y_val)),
            "test_samples": int(len(y_test)),
            "val_positive": int(y_val.sum()),
            "test_positive": int(y_test.sum()),
        },
        "domain_shift": {
            "domain_classifier_auc": domain_auc,
            "pc1_var_ratio": float(var_ratio[0]),
            "pc2_var_ratio": float(var_ratio[1]),
            "pc1_mean_gap": pc_mean_gap[0],
            "pc2_mean_gap": pc_mean_gap[1],
        },
        "plot": plot_path,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report saved to: {args.output}")


if __name__ == "__main__":
    main()
