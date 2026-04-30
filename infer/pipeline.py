"""
Simple high-level API for V-JEPA 2.1.

Usage:

    from vjepa21 import VJEPA21

    # Feature extraction
    model = VJEPA21.load("path/to/model", device="cuda")
    features = model.encode("video.mp4")
    features = model.encode(numpy_array)           # (F, H, W, C) uint8
    features = model.encode(torch_tensor)          # (F, C, H, W) float

    # Classification
    model = VJEPA21.load("path/to/model", task="classification", num_labels=2,
                         labels=["nonfight", "fight"])
    pred = model.predict("video.mp4")              # {"fight": 0.92, "nonfight": 0.08}

    # Train classifier
    model.train_classifier(
        train_data={"fight": "path/to/fight/", "nonfight": "path/to/nonfight/"},
        val_split=0.15,
        epochs=10,
        batch_size=8,
        num_frames=64,
        output_dir="checkpoints/",
    )

    # Save / load trained model
    model.save("my_model/")
    model = VJEPA21.load("my_model/", task="classification")
"""

import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .configuration import VJEPA21Config  # type: ignore
from .modeling import VJEPA21ForVideoClassification, VJEPA21Model  # type: ignore


def _load_video_decord(path, num_frames=64):
    """Load video frames using decord. Returns (F, C, H, W) float32 in [0,1]."""
    import decord  # type: ignore

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(path), num_threads=1)
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {path}")
    indices = torch.linspace(0, total - 1, steps=num_frames).round().long()
    frames = vr.get_batch(indices.tolist())  # (F, H, W, C) uint8
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    return frames


def _load_video_cv2(path, num_frames=64):
    """Fallback: load video frames using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Empty video: {path}")
    indices = (
        torch.linspace(0, total - 1, steps=num_frames).round().long().tolist()
    )
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))
        else:
            frames.append(torch.zeros(1, 1, 3, dtype=torch.uint8))
    cap.release()
    frames = torch.stack(frames)  # (F, H, W, C)
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    return frames


def load_video(path, num_frames=64):
    """Load video from file path. Returns (F, C, H, W) float32 in [0,1]."""
    try:
        return _load_video_decord(path, num_frames)
    except ImportError:
        return _load_video_cv2(path, num_frames)


def preprocess_frames(frames, crop_size=384):
    """Resize and normalize frames. Input: (F, C, H, W) float [0,1]. Output: (F, C, H, W) float."""
    from torchvision.transforms import v2

    frames = v2.functional.resize(frames, [crop_size, crop_size])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames


class _VideoFolderDataset(Dataset):
    def __init__(self, samples, num_frames, crop_size, augment=False):
        self.samples = samples
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):  # type: ignore
        path, label = self.samples[idx]
        try:
            frames = load_video(path, self.num_frames)
            if self.augment:
                from torchvision.transforms import v2

                frames = (
                    v2.functional.horizontal_flip(frames)
                    if random.random() > 0.5
                    else frames
                )
            frames = preprocess_frames(frames, self.crop_size)
            return frames, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


class VJEPA21:
    """
    Simple high-level API for V-JEPA 2.1.

    Examples::

        # Feature extraction
        model = VJEPA21.load("path/to/model", device="cuda")
        features = model.encode("video.mp4")

        # Classification
        model = VJEPA21.load("path/to/model", task="classification",
                             num_labels=2, labels=["nonfight", "fight"])
        result = model.predict("video.mp4")
    """

    def __init__(self, model, config, device, task="encode", labels=None):
        self.model = model
        self.config = config
        self.device = device
        self.task = task
        self.labels = labels
        self.num_frames = config.frames_per_clip
        self.crop_size = config.crop_size

    @classmethod
    def load(
        cls,
        path,
        device="cuda",
        task="encode",
        num_labels=2,
        labels=None,
        dtype=torch.bfloat16,
    ):
        """
        Load a V-JEPA 2.1 model.

        Args:
            path: Path to HF-format model directory (config.json + model.safetensors)
            device: "cuda", "cpu", or torch.device
            task: "encode" for feature extraction, "classification" for video classification
            num_labels: Number of classes (only for task="classification")
            labels: List of label names, e.g. ["nonfight", "fight"]
            dtype: Model dtype (default bfloat16)

        Returns:
            VJEPA21 instance
        """
        device = torch.device(device)

        if task == "classification":
            model = VJEPA21ForVideoClassification.from_pretrained(
                path,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                torch_dtype=dtype,
            )
        else:
            model = VJEPA21Model.from_pretrained(path, torch_dtype=dtype)

        model = model.to(device).eval()
        config = model.config
        return cls(model, config, device, task=task, labels=labels)

    def _to_tensor(self, video_input):
        """Convert various input formats to (1, F, C, H, W) tensor."""
        if isinstance(video_input, (str, Path)):
            frames = load_video(str(video_input), self.num_frames)
        elif isinstance(video_input, np.ndarray):
            # Expect (F, H, W, C) uint8
            if video_input.ndim == 4 and video_input.shape[-1] in (1, 3):
                frames = (
                    torch.from_numpy(video_input).permute(0, 3, 1, 2).float()
                    / 255.0
                )
            elif video_input.ndim == 4 and video_input.shape[1] in (1, 3):
                frames = torch.from_numpy(video_input).float()
                if frames.max() > 1.0:
                    frames = frames / 255.0
            else:
                raise ValueError(
                    f"Unexpected numpy shape: {video_input.shape}. Expected (F,H,W,C) or (F,C,H,W)"
                )
        elif isinstance(video_input, torch.Tensor):
            frames = video_input.float()
            if frames.max() > 1.0:
                frames = frames / 255.0
            # Handle (F, H, W, C) format
            if frames.ndim == 4 and frames.shape[-1] in (1, 3):
                frames = frames.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported input type: {type(video_input)}")

        frames = preprocess_frames(frames, self.crop_size)
        return frames.unsqueeze(0)  # (1, F, C, H, W)

    @torch.no_grad()
    def encode(self, video_input, pool=False):
        """
        Extract features from a video.

        Args:
            video_input: file path (str), numpy array (F,H,W,C), or torch tensor (F,C,H,W)
            pool: If True, return mean-pooled features (1D vector). If False, return per-token features.

        Returns:
            torch.Tensor: features. Shape (1, num_tokens, hidden_size) or (1, hidden_size) if pool=True.
        """
        tensor = self._to_tensor(video_input).to(
            self.device, dtype=self.model.dtype
        )

        if self.task == "classification":
            out = self.model.vjepa2_1(
                pixel_values_videos=tensor, skip_predictor=True
            )
        else:
            out = self.model(pixel_values_videos=tensor, skip_predictor=True)

        features = out.last_hidden_state
        if pool:
            features = features.mean(dim=1)
        return features.cpu()

    @torch.no_grad()
    def predict(self, video_input):
        """
        Classify a video.

        Args:
            video_input: file path (str), numpy array (F,H,W,C), or torch tensor (F,C,H,W)

        Returns:
            dict: {label: probability} sorted by probability descending.
            If no labels provided, keys are integer indices.
        """
        if self.task != "classification":
            raise ValueError(
                "predict() requires task='classification'. Use encode() for feature extraction."
            )

        tensor = self._to_tensor(video_input).to(
            self.device, dtype=self.model.dtype
        )
        outputs = self.model(pixel_values_videos=tensor)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu()

        if self.labels:
            result = {
                label: probs[i].item() for i, label in enumerate(self.labels)
            }
        else:
            result = {i: probs[i].item() for i in range(len(probs))}

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def train_classifier(
        self,
        train_data,
        val_split=0.15,
        epochs=10,
        batch_size=8,
        num_frames=None,
        lr=1e-4,
        weight_decay=1e-4,
        num_workers=4,
        output_dir="checkpoints",
        freeze_encoder=True,
    ):
        """
        Train a classification head on video folders.

        Args:
            train_data: dict mapping label_name -> folder_path, e.g.
                        {"fight": "/path/to/fight/", "nonfight": "/path/to/nonfight/"}
            val_split: Fraction of data for validation (default 0.15)
            epochs: Number of training epochs
            batch_size: Batch size
            num_frames: Frames per clip (default: config.frames_per_clip)
            lr: Learning rate
            weight_decay: Weight decay
            num_workers: DataLoader workers
            output_dir: Where to save checkpoints
            freeze_encoder: Whether to freeze the encoder (default True)

        Returns:
            dict with training history
        """
        if self.task != "classification":
            raise ValueError(
                "train_classifier() requires task='classification'"
            )

        num_frames = num_frames or self.num_frames
        self.labels = list(train_data.keys())
        device = self.device

        # Build samples: [(path, label_idx), ...]
        samples = []
        for label_idx, (label_name, folder) in enumerate(train_data.items()):
            videos = glob.glob(os.path.join(folder, "*.mp4"))
            videos += glob.glob(os.path.join(folder, "*.avi"))
            videos += glob.glob(os.path.join(folder, "*.mov"))
            print(f"  {label_name}: {len(videos)} videos (label={label_idx})")
            samples.extend([(v, label_idx) for v in sorted(videos)])

        random.shuffle(samples)
        val_size = int(len(samples) * val_split)
        train_samples, val_samples = samples[val_size:], samples[:val_size]
        print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

        train_ds = _VideoFolderDataset(
            train_samples, num_frames, self.crop_size, augment=True
        )
        val_ds = _VideoFolderDataset(
            val_samples, num_frames, self.crop_size, augment=False
        )

        def collate(batch):
            clips, labels = zip(*batch)
            return torch.stack(clips), torch.tensor(labels, dtype=torch.long)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

        # Freeze encoder
        if freeze_encoder:
            for p in self.model.vjepa2_1.parameters():
                p.requires_grad = False

        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        os.makedirs(output_dir, exist_ok=True)
        best_val_acc = 0.0
        history = []

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]"
            )
            for clips, labels in pbar:
                clips = clips.to(device, dtype=self.model.dtype)
                labels = labels.to(device)
                outputs = self.model(pixel_values_videos=clips, labels=labels)
                optimizer.zero_grad()
                outputs.loss.backward()
                optimizer.step()
                train_loss += outputs.loss.item() * clips.size(0)
                train_correct += (
                    (outputs.logits.argmax(-1) == labels).sum().item()
                )
                train_total += labels.size(0)
                pbar.set_postfix(
                    loss=f"{outputs.loss.item():.4f}",
                    acc=f"{train_correct / train_total:.4f}",
                )

            # Val
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for clips, labels in tqdm(val_loader, desc="[val]"):
                    clips = clips.to(device, dtype=self.model.dtype)
                    labels = labels.to(device)
                    outputs = self.model(
                        pixel_values_videos=clips, labels=labels
                    )
                    val_loss += outputs.loss.item() * clips.size(0)
                    val_correct += (
                        (outputs.logits.argmax(-1) == labels).sum().item()
                    )
                    val_total += labels.size(0)

            scheduler.step()

            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": train_loss / train_total,
                "train_acc": train_correct / train_total,
                "val_loss": val_loss / val_total if val_total > 0 else 0,
                "val_acc": val_correct / val_total if val_total > 0 else 0,
            }
            history.append(epoch_stats)
            print(
                f"Epoch {epoch + 1}: train_acc={epoch_stats['train_acc']:.4f} val_acc={epoch_stats['val_acc']:.4f}"
            )

            if epoch_stats["val_acc"] > best_val_acc:
                best_val_acc = epoch_stats["val_acc"]
                self.save(os.path.join(output_dir, "best"))
                print(f"  Saved best (val_acc={best_val_acc:.4f})")

        print(f"\nDone. Best val_acc: {best_val_acc:.4f}")
        return history

    def save(self, path):
        """Save model + config + labels."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        if self.labels:
            import json

            with open(os.path.join(path, "labels.json"), "w") as f:
                json.dump(self.labels, f)

    @classmethod
    def from_trained(cls, path, device="cuda", dtype=torch.bfloat16):
        """Load a previously saved trained model (auto-detects task from saved files)."""
        import json

        labels_path = os.path.join(path, "labels.json")
        labels = None
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                labels = json.load(f)

        config = VJEPA21Config.from_pretrained(path)
        has_num_labels = (
            hasattr(config, "num_labels") and config.num_labels > 0
        )

        if has_num_labels and labels:
            return cls.load(
                path,
                device=device,
                task="classification",
                num_labels=config.num_labels,
                labels=labels,
                dtype=dtype,
            )
        else:
            return cls.load(path, device=device, task="encode", dtype=dtype)
