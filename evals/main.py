# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import os
import pprint
import random
from collections import defaultdict

import yaml

from clearml import Dataset as ClearMLDataset, InputModel, Task

from evals.scaffold import main as eval_main  # type: ignore
from src.utils.distributed import init_distributed  # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument(
    "--val_only", action="store_true", help="only run eval", default=False
)
parser.add_argument(
    "--fname",
    type=str,
    help="name of config file to load",
    default="configs.yaml",
)
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0"],
    help="which devices to use on local machine",
)
parser.add_argument("--project", type=str, default="v-jepa-2.1")
parser.add_argument(
    "--task_name",
    type=str,
    default=None,
    help="ClearML task name (defaults to config eval_name)",
)
parser.add_argument(
    "--remote",
    action="store_true",
    default=False,
    help="Execute on a remote ClearML agent queue",
)
parser.add_argument(
    "--queue",
    type=str,
    default="default",
    help='ClearML agent queue name (default: "default")',
)
parser.add_argument(
    "--output_uri",
    type=str,
    default=None,
    help="ClearML output URI for model artifacts (e.g. s3://bucket/models)",
)
parser.add_argument(
    "--dataset_id",
    type=str,
    default="10b538da314e4a2d880c60b8a9f64935",
    help="ClearML dataset ID — downloads on remote agent and overrides "
    + "data paths in config",
)
parser.add_argument(
    "--model_id",
    type=str,
    default="e1c3e0025a5c401c84a263c8dc30d1d6",
    help="ClearML model ID — downloads pretrained model on remote agent "
    + "and overrides model_kwargs.checkpoint",
)
parser.add_argument(
    "--packages",
    type=str,
    nargs="*",
    default=None,
    help="Extra pip packages to install on remote agent",
)
parser.add_argument(
    "--snapshot_freq",
    type=int,
    default=5,
    help="Upload model snapshot to ClearML every N epochs "
    + "(default: 5, 0 to disable)",
)
parser.add_argument(
    "--debugmode",
    type=bool,
    default=False,
    help="Setting this to true will not spin up new processes. "
    "The main code runs the main process, which makes it easier to debug with checkpointing.",
)
parser.add_argument(
    "--folder",
    type=str,
    help="location to save logs",
    default="",
)
parser.add_argument("--override_config_folder", action="store_true")
parser.add_argument(
    "--checkpoint", type=str, help="location of pretrained ckpt"
)
parser.add_argument("--model_name", type=str, help="Model name")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--use_fsdp", action="store_true")
parser.add_argument(
    "--balance_classes",
    action="store_true",
    default=False,
    help="Auto-balance training/validation CSVs by undersampling "
    "the majority class to match the minority class",
)
parser.add_argument(
    "--balance_seed",
    type=int,
    default=42,
    help="Random seed for class balancing (default: 42)",
)


def _balance_csv(csv_path, seed=42):
    """Balance a V-JEPA CSV by undersampling the majority class.

    Reads '<video_path> <label>' lines, groups by label, undersamples
    the majority class to match the minority, shuffles, and writes a
    new file with '_balanced' suffix.

    Returns the path to the balanced CSV, or None on failure.
    """
    # Parse entries
    entries_by_class = defaultdict(list)
    delimiter = " "
    with open(csv_path, "r") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            delimiter = "," if "," in line_stripped else " "
            parts = line_stripped.split(delimiter, 1)
            if len(parts) < 2:
                continue
            video_path = parts[0].strip()
            label = int(parts[1].strip())
            entries_by_class[label].append(video_path)

    if not entries_by_class:
        return None

    # Print original distribution
    class_counts = {
        cls: len(items) for cls, items in sorted(entries_by_class.items())
    }
    min_count = min(class_counts.values())
    print(f"  Balancing {csv_path}:")
    print(f"    Original: {class_counts}")
    print(f"    Target: {min_count} per class")

    # Undersample each class to min_count
    random.seed(seed)
    balanced = []
    for cls in sorted(entries_by_class.keys()):
        items = entries_by_class[cls]
        if len(items) > min_count:
            items = random.sample(items, min_count)
        balanced.extend((path, cls) for path in items)

    # Shuffle
    random.shuffle(balanced)

    # Write balanced CSV
    base, ext = os.path.splitext(csv_path)
    balanced_path = f"{base}_balanced{ext}"
    with open(balanced_path, "w") as f:
        for video_path, label in balanced:
            f.write(f"{video_path}{delimiter}{label}\n")

    balanced_counts = defaultdict(int)
    for _, label in balanced:
        balanced_counts[label] += 1
    print(f"    Balanced: {dict(sorted(balanced_counts.items()))}")
    print(f"    Saved: {balanced_path}")
    return balanced_path


def process_main(args, rank, fname, world_size, devices):
    import logging
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        if args.val_only:
            params["val_only"] = True

        if args.checkpoint:
            params["model_kwargs"]["checkpoint"] = args.checkpoint

        if args.model_name:
            params["model_kwargs"]["pretrain_kwargs"]["encoder"][
                "model_name"
            ] = args.model_name

        if args.batch_size:
            params["experiment"]["optimization"]["batch_size"] = (
                args.batch_size
            )

        if args.override_config_folder:
            params["folder"] = args.folder
        params["use_fsdp"] = args.use_fsdp
        logger.info("loaded params...")

    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the eval with loaded config
    eval_main(params["eval_name"], args_eval=params)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load config early for ClearML
    with open(args.fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    # Inject CLI-only args into params
    params["snapshot_freq"] = args.snapshot_freq

    # Initialize ClearML task (before spawning workers so all logs are captured)
    clearml_task_name = args.task_name or params.get(
        "eval_name", "vjepa2-eval"
    )
    init_kwargs = dict(
        project_name=args.project,
        task_name=clearml_task_name,
        task_type=Task.TaskTypes.training,
        output_uri=args.output_uri or "s3://ai-dataset-clearml/clearml/models",
        auto_connect_frameworks={'pytorch': False},
    )

    task = Task.init(**init_kwargs)

    # Use forked repo via HTTPS (agent can clone without SSH keys)
    task.set_repo(repo="https://github.com/tinnapopd/vjepa2.git")
    task.connect(params)

    # Set required packages for remote execution
    if args.packages:
        task.set_packages(packages=args.packages)
    elif os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            pkgs = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        if "clearml" not in pkgs:
            pkgs.append("clearml")
        task.set_packages(packages=pkgs)

    # Remote execution: enqueue and exit locally
    if args.remote:
        task.execute_remotely(queue_name=args.queue, exit_process=True)

    # -------------------------------------------------------------------
    # Everything below runs on the REMOTE agent (or locally if --remote
    # was not passed)
    # -------------------------------------------------------------------

    # If a ClearML dataset ID is specified, download it and override data paths
    if args.dataset_id:
        print(f"Downloading ClearML dataset: {args.dataset_id}")
        dataset = ClearMLDataset.get(dataset_id=args.dataset_id)
        dataset_path = dataset.get_local_copy()
        print(f"Dataset downloaded to: {dataset_path}")

        # Override data paths in config with downloaded dataset
        if "experiment" in params and "data" in params["experiment"]:
            data_cfg = params["experiment"]["data"]
            train_csv = data_cfg.get("dataset_train", "")
            val_csv = data_cfg.get("dataset_val", "")

            train_basename = (
                os.path.basename(train_csv) if train_csv else "train.csv"
            )
            val_basename = os.path.basename(val_csv) if val_csv else "val.csv"

            new_train = os.path.join(dataset_path, train_basename)
            new_val = os.path.join(dataset_path, val_basename)

            # Rewrite CSV files: replace relative video paths with absolute paths
            for csv_path in [new_train, new_val]:
                if not os.path.exists(csv_path):
                    continue
                with open(csv_path, "r") as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    delimiter = "," if "," in line_stripped else " "
                    parts = line_stripped.split(delimiter)
                    if len(parts) >= 2:
                        video_rel_path = parts[0]
                        video_abs_path = os.path.join(
                            dataset_path,
                            os.path.basename(os.path.dirname(video_rel_path)),
                            os.path.basename(video_rel_path),
                        )
                        if not os.path.exists(video_abs_path):
                            # Walk the dataset dir to find the file
                            video_fname = os.path.basename(video_rel_path)
                            found = False
                            for root, dirs, files in os.walk(dataset_path):
                                if video_fname in files:
                                    video_abs_path = os.path.join(
                                        root, video_fname
                                    )
                                    found = True
                                    break
                            if not found:
                                video_abs_path = (
                                    video_rel_path  # keep original
                                )
                        new_lines.append(
                            f"{video_abs_path}{delimiter}{delimiter.join(parts[1:])}\n"
                        )
                    else:
                        new_lines.append(line)
                with open(csv_path, "w") as f:
                    f.writelines(new_lines)
                print(f"  Rewrote video paths in {csv_path}")

            if os.path.exists(new_train):
                data_cfg["dataset_train"] = new_train
                print(f"  dataset_train -> {new_train}")
            if os.path.exists(new_val):
                data_cfg["dataset_val"] = new_val
                print(f"  dataset_val -> {new_val}")

    # Auto-balance training/validation CSVs if requested
    if args.balance_classes:
        if "experiment" in params and "data" in params["experiment"]:
            data_cfg = params["experiment"]["data"]
            for csv_key in ["dataset_train", "dataset_val"]:
                csv_path = data_cfg.get(csv_key, "")
                if not csv_path or not os.path.exists(csv_path):
                    continue
                balanced_path = _balance_csv(
                    csv_path, seed=args.balance_seed
                )
                if balanced_path:
                    data_cfg[csv_key] = balanced_path
                    print(f"  {csv_key} -> {balanced_path} (balanced)")

    # If a ClearML model ID is specified, download it and override pretrain paths
    if args.model_id:
        print(f"Downloading ClearML model: {args.model_id}")
        model = InputModel(model_id=args.model_id)
        model_path = model.get_local_copy()
        print(f"Model downloaded to: {model_path}")

        # Override model_kwargs.checkpoint in config
        if "model_kwargs" in params:
            params["model_kwargs"]["checkpoint"] = model_path
            print(f"  model_kwargs.checkpoint -> {model_path}")

    # Re-write the config so spawned workers pick up the new paths
    config_dir = os.path.dirname(os.path.abspath(args.fname))
    updated_fname = os.path.join(config_dir, "_clearml_config.yaml")
    with open(updated_fname, "w") as f:
        yaml.dump(params, f)
    args.fname = updated_fname

    num_gpus = len(args.devices)
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    if args.debugmode:
        if args.use_fsdp:
            process_main(
                args=args,
                rank=int(os.environ["RANK"]),
                fname=args.fname,
                world_size=int(os.environ["WORLD_SIZE"]),
                devices=args.devices,
            )
        else:
            process_main(
                args=args,
                rank=0,
                fname=args.fname,
                world_size=1,
                devices=["cuda:0"],
            )
    else:
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=process_main,
                args=(args, rank, args.fname, num_gpus, args.devices),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
