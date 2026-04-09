#!/usr/bin/env python
# Upload a local dataset directory to ClearML.
#
# Usage:
#   python scripts/upload_dataset.py \
#     --dataset_dir /path/to/your/video/dataset \
#     --project "V-JEPA" \
#     --name "my-video-dataset"
#
import argparse
import os

from clearml import Dataset as ClearMLDataset


def clean_macos_artifacts(root_path: str) -> None:
    """Remove .DS_Store files and __MACOSX dirs before upload."""
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f == ".DS_Store":
                os.remove(os.path.join(root, f))
        if "__MACOSX" in dirs:
            macosx_path = os.path.join(root, "__MACOSX")
            try:
                os.rmdir(macosx_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Upload a dataset to ClearML")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Absolute path to the local dataset directory",
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
        default=None,
        help="ClearML dataset name (default: directory basename)",
    )
    parser.add_argument(
        "--output_uri",
        type=str,
        default=None,
        help="S3/GCS URI for dataset storage (default: ClearML file server)",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dataset_name = args.name or os.path.basename(dataset_dir)

    print(f"Creating ClearML dataset: {dataset_name}")
    print(f"  Project : {args.project}")
    print(f"  Source  : {dataset_dir}")

    # Clean macOS artifacts
    clean_macos_artifacts(dataset_dir)

    # Create and upload
    create_kwargs = dict(
        dataset_project=args.project,
        dataset_name=dataset_name,
    )
    if args.output_uri:
        create_kwargs["output_uri"] = args.output_uri

    dataset = ClearMLDataset.create(**create_kwargs)
    dataset.add_files(path=dataset_dir, recursive=True)

    print("Uploading...")
    dataset.upload(verbose=True)
    dataset.finalize()

    print(f"\nDone! Dataset ID: {dataset.id}")
    print(f"Use this ID with:  --dataset_id {dataset.id}")


if __name__ == "__main__":
    main()
