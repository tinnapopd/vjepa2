# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader  # type: ignore
from transformers import AutoModel, AutoVideoProcessor  # type: ignore

import src.datasets.utils.video.transforms as video_transforms  # type: ignore
import src.datasets.utils.video.volume_transforms as volume_transforms  # type: ignore
from src.models.attentive_pooler import AttentiveClassifier  # type: ignore
from src.models.vision_transformer import vit_giant_xformers_rope  # type: ignore

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 encoder
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(
        pretrained_weights, weights_only=True, map_location="cpu"
    )["encoder"]
    pretrained_dict = {
        k.replace("module.", ""): v for k, v in pretrained_dict.items()
    }
    pretrained_dict = {
        k.replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            pretrained_weights, msg
        )
    )


def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 classifier
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(
        pretrained_weights, weights_only=True, map_location="cpu"
    )["classifiers"][0]
    pretrained_dict = {
        k.replace("module.", ""): v for k, v in pretrained_dict.items()
    }
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            pretrained_weights, msg
        )
    )


def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    # Eval transform has no random cropping nor flip
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            ),
        ]
    )
    return eval_transform


def get_video(video_path: str):
    vr = VideoReader(video_path)
    total_frames = len(vr)
    # Load all frames
    frame_idx = np.arange(total_frames)
    video = vr.get_batch(frame_idx).asnumpy()
    print(f"Loaded {total_frames} frames from {video_path}")
    return video


def forward_vjepa_video(
    video_path,
    model_hf,
    model_pt,
    hf_transform,
    pt_transform,
):
    # Run a sample inference with VJEPA
    with torch.inference_mode():
        # Read and pre-process the video
        video = get_video(video_path)  # T x H x W x C
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        x_pt = pt_transform(video).cuda().unsqueeze(0)
        x_hf = hf_transform(video, return_tensors="pt")[
            "pixel_values_videos"
        ].to("cuda")
        # Extract the patch-wise features from the last layer
        out_patch_features_pt = model_pt(x_pt)
        out_patch_features_hf = model_hf.get_vision_features(x_hf)

    return out_patch_features_hf, out_patch_features_pt


def get_vjepa_video_classification_results(
    classifier,
    out_patch_features_pt,
    class_names: list[str] | None = None,
):
    if class_names is None:
        class_names = ["background", "weaponized"]

    with torch.inference_mode():
        out_classifier = classifier(out_patch_features_pt)

    print(f"Classifier output shape: {out_classifier.shape}")

    num_classes = out_classifier.shape[-1]
    probs = (
        F.softmax(out_classifier, dim=-1)[0] * 100.0
    )  # convert to percentage
    top_k = min(num_classes, len(class_names))
    top_indices = out_classifier.topk(top_k).indices[0]

    print("Predicted class probabilities:")
    for idx in top_indices:
        i = idx.item()
        label = class_names[i] if i < len(class_names) else f"class_{i}"
        print(f"  {label}: {probs[i]:.2f}%")

    return


def run_inference(
    video_path: str,
    pt_model_path: str,
    classifier_model_path: str | None = None,
    num_classes: int = 174,
):
    # HuggingFace model repo name
    hf_model_name = "facebook/vjepa2-vitg-fpc64-384"

    # Validate video path exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Validate encoder checkpoint exists
    if not os.path.exists(pt_model_path):
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {pt_model_path}"
        )

    print(f"Video path: {video_path}")
    print(f"Encoder checkpoint: {pt_model_path}")

    # Initialize the HuggingFace model, load pretrained weights
    model_hf = AutoModel.from_pretrained(hf_model_name)
    model_hf.cuda().eval()

    # Build HuggingFace preprocessing transform
    hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)
    img_size = hf_transform.crop_size["height"]  # E.g. 384, 256, etc.

    # Initialize the PyTorch model, load pretrained weights
    model_pt = vit_giant_xformers_rope(
        img_size=(img_size, img_size), num_frames=64
    )
    model_pt.cuda().eval()
    load_pretrained_vjepa_pt_weights(model_pt, pt_model_path)

    # Build PyTorch preprocessing transform
    pt_video_transform = build_pt_video_transform(img_size=img_size)

    # Inference on video
    out_patch_features_hf, out_patch_features_pt = forward_vjepa_video(
        video_path, model_hf, model_pt, hf_transform, pt_video_transform
    )

    print(
        f"""
        Inference results on video:
        HuggingFace output shape: {out_patch_features_hf.shape}
        PyTorch output shape:     {out_patch_features_pt.shape}
        Absolute difference sum:  {torch.abs(out_patch_features_pt - out_patch_features_hf).sum():.6f}
        Close: {torch.allclose(out_patch_features_pt, out_patch_features_hf, atol=1e-3, rtol=1e-3)}
        """
    )

    # Run classification if a classifier checkpoint is provided
    if classifier_model_path is not None:
        if not os.path.exists(classifier_model_path):
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {classifier_model_path}"
            )
        print(f"Classifier checkpoint: {classifier_model_path}")

        classifier = (
            AttentiveClassifier(
                embed_dim=model_pt.embed_dim,
                num_heads=16,
                depth=4,
                num_classes=num_classes,
            )
            .cuda()
            .eval()
        )
        load_pretrained_vjepa_classifier_weights(
            classifier, classifier_model_path
        )

        get_vjepa_video_classification_results(
            classifier, out_patch_features_pt
        )
    else:
        print("No classifier checkpoint provided — skipping classification.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="V-JEPA 2.1 video inference evaluation"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        default="pretrained-models/vjepa2_1_vitG_384.pt",
        help="Path to the V-JEPA 2.1 encoder checkpoint (default: pretrained-models/vjepa2_1_vitG_384.pt)",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default=None,
        help="Path to the attentive probe checkpoint (optional)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classification classes (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        video_path=args.video_path,
        pt_model_path=args.encoder_checkpoint,
        classifier_model_path=args.classifier_checkpoint,
        num_classes=args.num_classes,
    )
