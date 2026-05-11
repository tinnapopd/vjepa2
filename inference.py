import argparse
import os
import csv
import time
import warnings
from typing import List, Tuple, Dict, Any

import decord  # type: ignore
import torch
import torch.nn.functional as F

import src.datasets.utils.video.transforms as video_transforms  # type: ignore
import src.datasets.utils.video.volume_transforms as volume_transforms  # type: ignore

warnings.filterwarnings("ignore", category=FutureWarning)


class EvalDataset:
    def __init__(self, dataset_dir: str):
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError("Dataset directory not found.")

        self.dataset_dir = dataset_dir
        self.dataset_classes = self._get_classes()
        self.dataset_class_index = {
            classname: i for i, classname in enumerate(self.dataset_classes)
        }

    def _get_classes(self):
        classes = [
            dir_name
            for dir_name in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, dir_name))
        ]
        classes.sort()
        return classes

    def get_paths(self) -> Tuple[List[str], List[int], List[str]]:
        videos_path = []
        video_classes = []
        labels_path = []
        for class_name, class_index in self.dataset_class_index.items():
            video_dir = os.path.join(self.dataset_dir, class_name, "videos")
            for filename in os.listdir(video_dir):
                if not filename.endswith((".mp4", ".avi", ".mkv")):
                    continue

                video_path = os.path.join(video_dir, filename)
                filename, _ = os.path.splitext(filename)
                label_path = os.path.join(
                    self.dataset_dir, class_name, "labels", f"{filename}.csv"
                )
                videos_path.append(video_path)
                video_classes.append(class_index)
                labels_path.append(label_path)

        return videos_path, video_classes, labels_path

    @staticmethod
    def load_labels(label_path: str) -> List[Tuple[float, float]]:
        segments = []
        if not os.path.isfile(label_path):
            return segments

        with open(label_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    start = float(row[0].strip())
                    end = float(row[1].strip())
                    segments.append((start, end))
                except ValueError:
                    continue
        return segments

    @staticmethod
    def clip_overlaps_any_label(
        clip_start_sec: float,
        clip_end_sec: float,
        labels: List[Tuple[float, float]],
    ) -> bool:
        for label_start, label_end in labels:
            if clip_start_sec <= label_end and clip_end_sec >= label_start:
                return True
        return False


def compute_metrics(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> Dict[str, float]:
    """Compute evaluation metrics from a confusion matrix."""
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if (4 * precision + recall) > 0
        else 0.0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "specificity": round(specificity, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "accuracy": round(accuracy, 4),
    }


def evaluate_clip_vjepa(
    frames,
    encoder,
    classifier,
    model_positive_idx: int,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    with torch.inference_mode():
        # frames should be in THWC format (list of HWC numpy arrays or
        # a numpy/torch tensor of shape [T, H, W, C]).
        # Ensure we have a list of numpy arrays for the transforms.
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()

        # Convert to list of HWC numpy arrays (what the transforms expect)
        frame_list = [frames[i] for i in range(frames.shape[0])]

        img_size = 384
        if hasattr(encoder, "patch_embed") and hasattr(
            encoder.patch_embed, "img_size"
        ):
            if isinstance(encoder.patch_embed.img_size, tuple):
                img_size = encoder.patch_embed.img_size[0]
            else:
                img_size = encoder.patch_embed.img_size

        # Match training eval transform exactly (from evals/video_classification_frozen/utils.py)
        short_side_size = int(256.0 / 224 * img_size)
        eval_transform = video_transforms.Compose(
            [
                video_transforms.Resize(
                    short_side_size, interpolation="bilinear"
                ),
                video_transforms.CenterCrop(size=(img_size, img_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        # Transform outputs CTHW tensor, add batch dim -> [1, C, T, H, W]
        x_pt = eval_transform(frame_list).to(device).unsqueeze(0)
        out_patch_features_pt = encoder(x_pt)
        out_classifier = classifier(out_patch_features_pt)

        probs = F.softmax(out_classifier[0], dim=0) * 100.0
        label_idx = out_classifier.argmax(dim=1).item()

        vjepa_label = label_idx
        vjepa_conf = probs[label_idx].item()
        vjepa_violent_conf = probs[model_positive_idx].item()

    return {
        "vjepa_label": vjepa_label,
        "vjepa_conf": vjepa_conf,
        "vjepa_violent_conf": vjepa_violent_conf,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V-JEPA 2.1 Batch Evaluation")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/tf/anomaly-detection/model-evaluation/dataset",
    )
    parser.add_argument(
        "--encoder_weights",
        type=str,
        default="pretrained-models/vjepa2_1_vitl_dist_vitG_384.pt",
        help="Path to V-JEPA encoder weights",
    )
    parser.add_argument(
        "--probe_weights",
        type=str,
        default="trained-probes/vitl-probe.pt",
        help="Path to attentive probe weights",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=4,
        help="Frame sampling stride (must match training config frame_step)",
    )
    parser.add_argument("--img_size", type=int, default=384, help="Image size")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weight_basename = os.path.basename(args.encoder_weights).lower()
    arch_part = (
        weight_basename.split("_")[2]
        if len(weight_basename.split("_")) > 2
        else ""
    )

    if "vitg" in arch_part:
        from src.models.vision_transformer import (  # type: ignore
            vit_gigantic_xformers as vit_model,
        )
    elif "vitl" in arch_part:
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
    pretrained_dict = torch.load(
        args.encoder_weights, map_location="cpu", weights_only=True
    )
    if "ema_encoder" in pretrained_dict:
        pretrained_dict = pretrained_dict["ema_encoder"]
    elif "target_encoder" in pretrained_dict:
        pretrained_dict = pretrained_dict["target_encoder"]
    elif "encoder" in pretrained_dict:
        pretrained_dict = pretrained_dict["encoder"]

    pretrained_dict = {
        k.replace("module.", ""): v for k, v in pretrained_dict.items()
    }
    pretrained_dict = {
        k.replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }
    encoder.load_state_dict(pretrained_dict, strict=False)
    encoder.to(device).eval()

    print("Loading Attentive Classifier...")
    from src.models.attentive_pooler import AttentiveClassifier  # type: ignore

    dataset = EvalDataset(args.dataset_dir)
    class_labels = dataset.dataset_classes

    probe_dict = torch.load(
        args.probe_weights, map_location="cpu", weights_only=True
    )
    if "classifiers" in probe_dict:
        probe_dict = probe_dict["classifiers"][0]

    # Strip DDP "module." prefix from checkpoint keys
    probe_dict = {k.replace("module.", ""): v for k, v in probe_dict.items()}

    # Dynamically determine number of classes from the probe weights
    model_num_classes = probe_dict["linear.weight"].shape[0]
    if model_num_classes == 2:
        model_positive_idx = 1  # weaponized
        print("Detected 2-class model. Classes: [background, weaponized]")
    elif model_num_classes == 3:
        model_positive_idx = 2  # weaponized (ignoring fighting)
        print("Detected 3-class model. Classes: [background, fighting, weaponized]. Ignoring fighting class.")
    else:
        model_positive_idx = model_num_classes - 1
        print(f"Detected {model_num_classes}-class model.")

    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=16,
        depth=4,
        num_classes=model_num_classes,
    )
    # Verify checkpoint loads correctly (catch silent mismatches)
    msg = classifier.load_state_dict(probe_dict, strict=False)
    if msg.missing_keys:
        print(f"WARNING: Missing keys in probe checkpoint: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(
            f"WARNING: Unexpected keys in probe checkpoint: {msg.unexpected_keys}"
        )
    if not msg.missing_keys and not msg.unexpected_keys:
        print("Probe checkpoint loaded successfully (all keys matched).")
    classifier.to(device).eval()

    videos_path, video_classes, labels_path = dataset.get_paths()

    tp, fp, tn, fn = 0, 0, 0, 0
    
    # Track metrics separately by folder
    folder_metrics = {}
    for c_name in class_labels:
        folder_metrics[c_name] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "videos": []}

    print(f"Dataset folder classes: {class_labels}")
    print(f"Frame sampling: {args.num_frames} frames with step {args.frame_step}"
          f" (covers {args.num_frames * args.frame_step} raw frames per clip)")
    print(f"Found {len(videos_path)} videos to evaluate.")
    print("Starting Batch Evaluation...")

    # Sort videos so we evaluate sequentially
    combined = list(zip(videos_path, video_classes, labels_path))
    combined.sort(key=lambda x: (x[1], x[0]))
    videos_path, video_classes, labels_path = zip(*combined)

    # Use numpy bridge so decord returns THWC numpy arrays
    # (matches what training dataloader produces for transforms)
    decord.bridge.set_bridge("native")
    for video_idx, (v_path, v_class, l_path) in enumerate(
        zip(videos_path, video_classes, labels_path)
    ):
        print(
            f"[{video_idx + 1}/{len(videos_path)}] Evaluating: {os.path.basename(v_path)}"
        )
        vid_tp, vid_fp, vid_tn, vid_fn = 0, 0, 0, 0
        try:
            vr = decord.VideoReader(v_path, ctx=decord.cpu(0))
            fps = vr.get_avg_fps()
            if not fps or fps == 0:
                fps = 30.0
            total_frames = len(vr)
        except Exception as e:
            print(f"Error reading {v_path}: {e}")
            continue

        labels = dataset.load_labels(l_path)

        # Calculate the number of raw frames needed per clip
        # (matching training: num_frames sampled with frame_step stride)
        raw_frames_per_clip = args.num_frames * args.frame_step

        video_start_time = time.time()
        for start_idx in range(0, total_frames, raw_frames_per_clip):
            end_idx = start_idx + raw_frames_per_clip
            if end_idx > total_frames:
                break

            try:
                # Sample every frame_step-th frame to match training
                indices = list(range(start_idx, end_idx, args.frame_step))
                # Returns THWC numpy arrays (native bridge)
                clip_frames = vr.get_batch(indices).asnumpy()
            except Exception as e:
                print(
                    f"Error reading frames {start_idx}-{end_idx} in {v_path}: {e}"
                )
                continue
            clip_start_sec = start_idx / fps
            clip_end_sec = end_idx / fps

            is_violent_clip = dataset.clip_overlaps_any_label(
                clip_start_sec, clip_end_sec, labels
            )

            res = evaluate_clip_vjepa(
                clip_frames, encoder, classifier, model_positive_idx, device
            )
            pred_label = res["vjepa_label"]

            is_pred_positive = (pred_label == model_positive_idx)
            is_true_positive = is_violent_clip

            if is_pred_positive and is_true_positive:
                tp += 1
                vid_tp += 1
            elif is_pred_positive and not is_true_positive:
                fp += 1
                vid_fp += 1
            elif not is_pred_positive and not is_true_positive:
                tn += 1
                vid_tn += 1
            elif not is_pred_positive and is_true_positive:
                fn += 1
                vid_fn += 1

        video_end_time = time.time()
        total_frames_processed = (
            vid_tp + vid_fp + vid_tn + vid_fn
        ) * args.num_frames
        elapsed_time = video_end_time - video_start_time
        inf_fps = (
            total_frames_processed / elapsed_time if elapsed_time > 0 else 0.0
        )

        print(
            f"  -> Video Results: TP={vid_tp}, FP={vid_fp}, "
            + f"TN={vid_tn}, FN={vid_fn} | Inference FPS: {inf_fps:.2f}"
        )

        # Get the class folder name for this video
        class_name = os.path.basename(os.path.dirname(os.path.dirname(v_path)))
        if class_name in folder_metrics:
            folder_metrics[class_name]["tp"] += vid_tp
            folder_metrics[class_name]["fp"] += vid_fp
            folder_metrics[class_name]["tn"] += vid_tn
            folder_metrics[class_name]["fn"] += vid_fn
            folder_metrics[class_name]["videos"].append({
                "name": os.path.basename(v_path),
                "tp": vid_tp,
                "fp": vid_fp,
                "tn": vid_tn,
                "fn": vid_fn,
                "fps": inf_fps
            })

    print("\nGlobal Evaluation Results:")
    global_metrics = compute_metrics(tp, fp, tn, fn)
    for k, v in global_metrics.items():
        print(f"{k}: {v}")

    print(f"\nGlobal Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Save results separate by using folder
    out_base_dir = "inference"
    os.makedirs(out_base_dir, exist_ok=True)
    
    print(f"\nSaving folder-separated results to '{out_base_dir}/' ...")
    for c_name, m_dict in folder_metrics.items():
        if len(m_dict["videos"]) == 0:
            continue
            
        c_dir = os.path.join(out_base_dir, c_name)
        os.makedirs(c_dir, exist_ok=True)
        
        c_tp, c_fp, c_tn, c_fn = m_dict["tp"], m_dict["fp"], m_dict["tn"], m_dict["fn"]
        c_metrics = compute_metrics(c_tp, c_fp, c_tn, c_fn)
        
        # Save metrics
        metrics_csv = os.path.join(c_dir, "metrics.csv")
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["TP", c_tp])
            writer.writerow(["FP", c_fp])
            writer.writerow(["TN", c_tn])
            writer.writerow(["FN", c_fn])
            for k, v in c_metrics.items():
                writer.writerow([k, v])
                
        # Save video stats
        video_csv = os.path.join(c_dir, "video_results.csv")
        with open(video_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Video", "TP", "FP", "TN", "FN", "Inference_FPS"])
            for v_stat in m_dict["videos"]:
                writer.writerow([
                    v_stat["name"], 
                    v_stat["tp"], 
                    v_stat["fp"], 
                    v_stat["tn"], 
                    v_stat["fn"], 
                    round(v_stat["fps"], 2)
                ])
                
        print(f"  Saved results for '{c_name}' in {c_dir}/")
