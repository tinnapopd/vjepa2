# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import os
import pprint

import yaml
from clearml import Dataset as ClearMLDataset, InputModel
from clearml import Task

from evals.scaffold import main as eval_main
from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument(
    "--val_only", action="store_true", help="only run eval", default=False
)
parser.add_argument(
    "--fname",
    type=str,
    help="name of config file to load",
    default="configs/custom-vjepa2_1.yaml",
)
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=[
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
        "cuda:4",
        "cuda:5",
        "cuda:6",
        "cuda:7",
    ],
    help="which devices to use on local machine",
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
parser.add_argument("--project", type=str, default="v-jepa")
parser.add_argument(
    "--task_name",
    type=str,
    default="My-Remote-VJEPA-Training-Run",
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
    default="nsb",
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
    default="f5c1dbe6d6bb44f5a5e29963317fc6a1",
    help="ClearML dataset ID — downloads on remote agent and overrides "
    + "data paths in config",
)
parser.add_argument(
    "--packages",
    type=str,
    nargs="*",
    default=None,
    help="Extra pip packages to install on remote agent",
)
parser.add_argument(
    "--model_id",
    type=str,
    default="9103f10ca1884117b1ec85efee24073d",
    help="ClearML model ID — downloads pretrained model on remote agent "
    + "and overrides pretrain paths",
)
parser.add_argument(
    "--snapshot_freq",
    type=int,
    default=5,
    help="Upload model snapshot to ClearML every N epochs "
    + "(default: 5, 0 to disable)",
)


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
        params["snapshot_freq"] = getattr(args, "snapshot_freq", 0)
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

    # Initialize ClearML task (before spawning workers so all logs are captured)
    clearml_task_name = args.task_name or params.get("eval_name", "vjepa-eval")
    init_kwargs = dict(
        project_name=args.project,
        task_name=clearml_task_name,
        task_type=Task.TaskTypes.training,
        output_uri=args.output_uri or "s3://ai-dataset-clearml/clearml/models",
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

    # If a ClearML dataset ID is specified, download it and override data paths
    if args.dataset_id:
        print(f"Downloading ClearML dataset: {args.dataset_id}")
        dataset = ClearMLDataset.get(dataset_id=args.dataset_id)
        dataset_path = dataset.get_local_copy()
        print(f"Dataset downloaded to: {dataset_path}")

        # Override data paths in config with downloaded dataset
        if "data" in params:
            train_csv = params["data"].get("dataset_train", "")
            val_csv = params["data"].get("dataset_val", "")

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
                    parts = line.strip().split(" ")
                    if len(parts) >= 2:
                        video_rel_path = parts[0]
                        # Try to resolve the video path relative to the dataset
                        # download dir
                        video_abs_path = os.path.join(
                            dataset_path,
                            os.path.basename(os.path.dirname(video_rel_path)),
                            os.path.basename(video_rel_path),
                        )
                        if not os.path.exists(video_abs_path):
                            # Try walking the dataset dir to find the file
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
                            f"{video_abs_path} {' '.join(parts[1:])}\n"
                        )
                    else:
                        new_lines.append(line)
                with open(csv_path, "w") as f:
                    f.writelines(new_lines)
                print(f"  Rewrote video paths in {csv_path}")

            if os.path.exists(new_train):
                params["data"]["dataset_train"] = new_train
                print(f"  dataset_train -> {new_train}")
            if os.path.exists(new_val):
                params["data"]["dataset_val"] = new_val
                print(f"  dataset_val -> {new_val}")

        # Re-write the config so spawned workers pick up the new paths
        updated_fname = os.path.join(dataset_path, "_clearml_config.yaml")
        with open(updated_fname, "w") as f:
            yaml.dump(params, f)
        args.fname = updated_fname

    # If a ClearML model ID is specified, download it and override pretrain paths
    if args.model_id:
        print(f"Downloading ClearML model: {args.model_id}")
        model = InputModel(model_id=args.model_id)
        model_path = model.get_local_copy()
        print(f"Model downloaded to: {model_path}")

        # Override pretrain paths in config
        if "model_kwargs" in params:
            params["model_kwargs"]["checkpoint"] = model_path
            print(f"  model_kwargs.checkpoint -> {model_path}")

        # Re-write the config so spawned workers pick up the new paths
        config_dir = os.path.dirname(args.fname)
        if not os.path.isabs(args.fname) or config_dir == "":
            config_dir = os.getcwd()
        updated_fname = os.path.join(config_dir, "_clearml_config.yaml")
        with open(updated_fname, "w") as f:
            yaml.dump(params, f)
        args.fname = updated_fname

    if args.debugmode:
        # FSDP debugging (use torchrun)
        if args.use_fsdp:
            process_main(
                args=args,
                rank=int(os.environ["RANK"]),
                fname=args.fname,
                world_size=int(os.environ["WORLD_SIZE"]),
                devices=args.devices,
            )
        # Single-GPU debugging
        else:
            process_main(
                args=args,
                rank=0,
                fname=args.fname,
                world_size=1,
                devices=["cuda:0"],
            )
    else:
        num_gpus = len(args.devices)
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        for rank in range(num_gpus):
            mp.Process(
                target=process_main,
                args=(args, rank, args.fname, num_gpus, args.devices),
            ).start()
