#!/usr/bin/env python
# Upload a pretrained model to ClearML.
#
# Usage:
#   python scripts/upload_model.py \
#     --model_path pretrained/vjepa2_1_vitb_dist_vitG_384.pt \
#     --project "v-jepa" \
#     --name "vjepa2_1_vitb_dist_vitG_384-pretrained"
#
import argparse
import os

from clearml import OutputModel, Task


def main():
    parser = argparse.ArgumentParser(
        description="Upload a pretrained model to ClearML"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained/vjepa2_1_vitb_dist_vitG_384.pt",
        help="Path to the model file (default: pretrained/vjepa2_1_vitb_dist_vitG_384.pt)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="v-jepa",
        help="ClearML project name (default: v-jepa)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="vjepa2_1_vitb_dist_vitG_384-pretrained",
        help="Model name (default: vjepa2_1_vitb_dist_vitG_384-pretrained)",
    )
    parser.add_argument(
        "--output_uri",
        type=str,
        default=None,
        help="S3/GCS URI for model storage (default: ClearML file server)",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_name = args.name or os.path.splitext(os.path.basename(model_path))[0]

    # Create a task for model upload
    task = Task.init(
        project_name=args.project,
        task_name=f"Upload {model_name}",
        task_type=Task.TaskTypes.custom,
    )
    if task is None:
        raise RuntimeError(
            "Failed to initialize ClearML task. Check your ClearML configuration."
        )

    # Upload the model
    output_model = OutputModel(
        task=task,
        name=model_name,
        framework="PyTorch",
    )
    output_model.update_weights(
        weights_filename=model_path,
        auto_delete_file=False,
    )

    model_id = output_model.id
    print(f"\nDone! Model ID: {model_id}")
    print(f"Use this ID with:  --model_id {model_id}")

    task.close()


if __name__ == "__main__":
    main()
