import argparse
import csv
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import src.datasets.utils.video.transforms as video_transforms  # type: ignore
import src.datasets.utils.video.volume_transforms as volume_transforms  # type: ignore
from src.models.vision_transformer import vit_gigantic_xformers  # type: ignore

logger = logging.getLogger(__name__)


# CLI args


def add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--encoder_weight",
        type=str,
        default="/tf/data/pretrained-models/vjepa2_1_vitG_384.pt",
    )
    parser.add_argument(
        "--probe_weight",
        type=str,
        default="/tf/data/outputs/evals_2_1/vitG-384/weaponized_2cls/"
        "video_classification_frozen/weaponized-2cls-vitg16-384/best.pt",
    )
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_step", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument(
        "--yolo_base",
        type=str,
        default="/tf/data/pretrained-models/yolo26m.pt",
    )
    parser.add_argument("--human_threshold", type=float, default=0.3)
    parser.add_argument(
        "--yolo_weapon",
        type=str,
        default="/tf/data/pretrained-models/yolo26m-weapon-det.pt",
    )
    parser.add_argument("--weapon_threshold", type=float, default=0.41)
    parser.add_argument(
        "--yolo_violence",
        type=str,
        default="/tf/data/pretrained-models/yolo26m-cls.pt",
    )


# Model loading


def load_vjepa_encoder(
    weights_path: str,
    img_size: int,
    num_frames: int,
    device: str,
) -> torch.nn.Module:
    encoder = vit_gigantic_xformers(
        img_size=img_size,
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )
    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    for key in ("ema_encoder", "target_encoder", "encoder"):
        if key in state_dict:
            state_dict = state_dict[key]
            break

    state_dict = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in state_dict.items()
    }
    encoder.load_state_dict(state_dict, strict=False)
    encoder.to(device).eval()
    return encoder


def inspect_probe_metadata(weights_path: str) -> Tuple[int, int]:
    probe_dict = torch.load(
        weights_path, map_location="cpu", weights_only=True
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
    return num_classes, positive_idx


@dataclass(frozen=True)
class PipelineModels:
    encoder: torch.nn.Module
    num_classes: int
    positive_idx: int
    cls_model: YOLO
    human_model: YOLO
    weapon_model: YOLO


def load_pipeline_models(
    args: argparse.Namespace,
    device: str,
    cls_checkpoint: str,
) -> PipelineModels:
    logger.info("Loading V-JEPA 2.1 encoder …")
    encoder = load_vjepa_encoder(
        args.encoder_weight,
        img_size=args.img_size,
        num_frames=args.num_frames,
        device=device,
    )

    logger.info("Reading probe metadata …")
    num_classes, positive_idx = inspect_probe_metadata(args.probe_weight)
    logger.info(f"{num_classes}-class probe, positive_idx={positive_idx}")

    logger.info(f"Loading YOLO-CLS: {cls_checkpoint} …")
    cls_model = YOLO(cls_checkpoint)

    logger.info(f"Loading YOLO base (human det): {args.yolo_base} …")
    human_model = YOLO(args.yolo_base)

    logger.info(f"Loading YOLO weapon det: {args.yolo_weapon} …")
    weapon_model = YOLO(args.yolo_weapon)

    return PipelineModels(
        encoder=encoder,
        num_classes=num_classes,
        positive_idx=positive_idx,
        cls_model=cls_model,
        human_model=human_model,
        weapon_model=weapon_model,
    )


# Video I/O


def iter_clips_from_video(
    video_path: str,
    num_frames: int,
    frame_step: int,
) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray], float, float]]:
    """Yield (bgr_frames, rgb_frames, start_sec, end_sec) per clip.

    Stops at the first short read past the end of the video.
    """
    raw_per_clip = num_frames * frame_step
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open: {video_path}")
        return
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for cs in range(0, total_frames, raw_per_clip):
            ce = cs + raw_per_clip - 1
            if ce >= total_frames:
                break

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

            yield bgr, rgb, cs / fps, ce / fps
    finally:
        cap.release()


# Detectors


def has_human_in_clip(
    frames: List[np.ndarray],
    human_model: YOLO,
    human_threshold: float = 0.3,
    min_human_frames: int = 1,
) -> Tuple[bool, Dict[str, float]]:
    human_frame_count = 0
    max_humans_in_frame = 0

    for frame in frames:
        results = human_model.predict(
            frame,
            verbose=False,
            conf=human_threshold,
            classes=[0],
        )
        count = 0
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                count += len(r.boxes)
        if count > 0:
            human_frame_count += 1
            max_humans_in_frame = max(max_humans_in_frame, count)

    total = len(frames) if frames else 1
    detected = human_frame_count >= min_human_frames
    return detected, {
        "human_detected": 1.0 if detected else 0.0,
        "human_frame_ratio": human_frame_count / total,
        "human_max_count": float(max_humans_in_frame),
    }


def detect_weapons_in_clip(
    frames: List[np.ndarray],
    weapon_model: YOLO,
    weapon_threshold: float = 0.41,
) -> Tuple[bool, Dict[str, float]]:
    """Run weapon detection on clip frames and return stats."""
    weapon_frame_count = 0
    max_weapons_in_frame = 0
    max_conf = 0.0
    all_confs: List[float] = []

    for frame in frames:
        results = weapon_model.predict(
            frame,
            verbose=False,
            conf=weapon_threshold,
        )
        count = 0
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                count += len(r.boxes)
                for box in r.boxes:
                    c = float(box.conf[0])
                    all_confs.append(c)
                    max_conf = max(max_conf, c)
        if count > 0:
            weapon_frame_count += 1
            max_weapons_in_frame = max(max_weapons_in_frame, count)

    total = len(frames) if frames else 1
    detected = weapon_frame_count > 0
    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0

    return detected, {
        "weapon_detected": 1.0 if detected else 0.0,
        "weapon_frame_ratio": weapon_frame_count / total,
        "weapon_max_count": float(max_weapons_in_frame),
        "weapon_max_conf": max_conf,
        "weapon_mean_conf": mean_conf,
    }


# Feature extraction


def extract_vjepa_embeddings(
    frames,
    encoder: torch.nn.Module,
    device: str,
    pool: bool = True,
) -> torch.Tensor:
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
        patch_features = encoder(x)  # [1, num_patches, embed_dim]

        if pool:
            return patch_features[0].mean(dim=0)  # [embed_dim]
        return patch_features[0]  # [num_patches, embed_dim]


def extract_yolo_cls_embeddings(
    frames: List[np.ndarray],
    cls_model: YOLO,
    pool: str = "avg",
) -> np.ndarray:
    frame_embeds: List[np.ndarray] = []
    for frame in frames:
        # embed=[-2] extracts from the penultimate layer
        results = cls_model.predict(frame, verbose=False, embed=[-2])
        for r in results:
            if hasattr(r, "embeddings") and r.embeddings is not None:
                # r.embeddings is a list of tensors; take the first
                emb = r.embeddings[0]
                if hasattr(emb, "cpu"):
                    emb = emb.cpu().numpy()
                frame_embeds.append(emb.flatten())

    if not frame_embeds:
        # Fallback: return zeros if no embeddings could be extracted
        logger.warning("No YOLO-CLS embeddings extracted, returning zeros")
        return np.zeros(512, dtype=np.float32)

    stacked = np.stack(frame_embeds, axis=0)
    if pool == "max":
        return stacked.max(axis=0).astype(np.float32)
    return stacked.mean(axis=0).astype(np.float32)


def extract_embedding_features(
    rgb: List[np.ndarray],
    bgr: List[np.ndarray],
    encoder: torch.nn.Module,
    cls_model: YOLO,
    device: str,
    weapon_model: Optional[YOLO] = None,
    weapon_threshold: float = 0.41,
) -> np.ndarray:
    # V-JEPA embedding (pooled)
    vjepa_emb = extract_vjepa_embeddings(rgb, encoder, device, pool=True)
    vjepa_np = vjepa_emb.cpu().numpy().astype(np.float32)

    # YOLO-CLS embedding (pooled across frames)
    yolo_np = extract_yolo_cls_embeddings(bgr, cls_model, pool="mean")

    parts = [vjepa_np, yolo_np]

    # Weapon detection features
    if weapon_model is not None:
        _, weapon_stats = detect_weapons_in_clip(
            bgr, weapon_model, weapon_threshold
        )
        weapon_feat = np.array(list(weapon_stats.values()), dtype=np.float32)
        parts.append(weapon_feat)

    return np.concatenate(parts)


# Per-clip pipeline + output


def collect_clip_features(
    video_entries: Iterable[Tuple[str, Any]],
    models: PipelineModels,
    device: str,
    *,
    num_frames: int,
    frame_step: int,
    human_threshold: float,
    weapon_threshold: float,
    label_fn: Callable[[Any, float, float], int],
    metadata_fn: Callable[[str, Any, float, float, int], Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    metadata: List[Dict[str, Any]] = []
    skipped_no_human = 0

    for vp, ctx in video_entries:
        for bgr, rgb, cs_sec, ce_sec in iter_clips_from_video(
            vp, num_frames=num_frames, frame_step=frame_step
        ):
            has_human, _ = has_human_in_clip(
                bgr, models.human_model, human_threshold
            )
            if not has_human:
                skipped_no_human += 1
                continue

            emb = extract_embedding_features(
                rgb,
                bgr,
                models.encoder,
                models.cls_model,
                device,
                weapon_model=models.weapon_model,
                weapon_threshold=weapon_threshold,
            )
            gt = label_fn(ctx, cs_sec, ce_sec)

            all_features.append(emb)
            all_labels.append(gt)
            metadata.append(metadata_fn(vp, ctx, cs_sec, ce_sec, gt))

    if skipped_no_human:
        logger.info(f"Skipped {skipped_no_human} clips (no human detected)")

    X = np.stack(all_features, axis=0)
    y = np.array(all_labels, dtype=np.int32)
    logger.info(
        f"Collected {len(y)} clips — {int(y.sum())} positive, "
        f"{len(y) - int(y.sum())} negative, embedding dim = {X.shape[1]}"
    )
    return X, y, metadata


def write_clips_csv(path: str, metadata: List[Dict[str, Any]]) -> None:
    if not metadata:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metadata[0].keys()))
        w.writeheader()
        w.writerows(metadata)


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
