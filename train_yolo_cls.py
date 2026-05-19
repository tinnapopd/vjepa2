"""YOLO-CLS Violence Classifier Training Script.

Trains a YOLOv26m-cls model on a violence classification dataset.
The user provides the dataset in YOLO classification format:

    dataset-violence-cls/
    ├── train/
    │   ├── violence/
    │   └── non-violence/
    └── val/
        ├── violence/
        └── non-violence/

Usage:
    python train_yolo_cls.py \\
        --data /path/to/dataset-violence-cls \\
        --base-model yolo26m-cls.pt \\
        --epochs 50 \\
        --imgsz 224 \\
        --output trained-models/yolo26m-violence-cls
"""

import argparse
import logging
import os

from ultralytics import YOLO

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO-CLS violence classifier."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset root (YOLO classification format)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo26m-cls.pt",
        help="Pretrained YOLO classification checkpoint to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=str,
        default="trained-models/yolo26m-violence-cls",
        help="Project name / output directory for runs",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    # Validate dataset
    for split in ("train", "val"):
        p = os.path.join(args.data, split)
        if not os.path.isdir(p):
            raise SystemExit(f"Error: expected split directory not found: {p}")
        classes = [
            d for d in os.listdir(p)
            if os.path.isdir(os.path.join(p, d))
        ]
        logger.info(f"  {split}/: classes = {sorted(classes)}")

    logger.info(f"Loading base model: {args.base_model}")
    model = YOLO(args.base_model)

    logger.info("Starting training …")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        workers=args.workers,
        device=args.device,
        project=args.output,
        name="train",
        exist_ok=True,
        verbose=True,
    )

    logger.info(f"Training complete. Results saved to: {args.output}/train/")


if __name__ == "__main__":
    main()
