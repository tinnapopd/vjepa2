from __future__ import annotations

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig  # type: ignore
from transformers.utils import auto_docstring  # type: ignore


@auto_docstring
@strict
class VJEPA21Config(PreTrainedConfig):
    r"""
    crop_size (`int`, *optional*, defaults to 384):
        Input resolution of the model
    frames_per_clip (`int`, *optional*, defaults to 64):
        The number of frames the model has been pretrained with. Does not impact inference.
    tubelet_size (`int`, *optional*, defaults to 2):
        The number of temporal frames used for a single rastor, check paper for more information.
    img_temporal_dim_size (`int`, *optional*, defaults to 1):
        The temporal dimension size for image tokenization. Used to detect whether input is an image or video.
    modality_embedding (`bool`, *optional*, defaults to `True`):
        Whether to use separate image/video modality embeddings.
    n_output_distillation (`int`, *optional*, defaults to 4):
        Number of hierarchical layer outputs for deep self-supervision.
    interpolate_rope (`bool`, *optional*, defaults to `True`):
        Whether to use RoPE interpolation for variable resolution support.
    pretrained_crop_size (`int`, *optional*, defaults to 256):
        The crop size used during pretraining. Used to compute the pretrained grid size for RoPE interpolation.
    teacher_embed_dim (`int` or `None`, *optional*, defaults to `None`):
        Teacher embedding dimension for distilled models. If None, uses hidden_size.
    pred_has_context_proj (`bool`, *optional*, defaults to `True`):
        Whether the predictor uses a separate context projection.
    num_pooler_layers (`int`, *optional*, defaults to 3):
        The number of self-attention layers in the pooler.
    pred_hidden_size (`int`, *optional*, defaults to 384):
        Dimensionality of the predictor layers
    pred_num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Predictor
    pred_num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Predictor
    pred_num_mask_tokens (`int`, *optional*, defaults to 8):
        Define the number of mask tokens to use in the Predictor
    pred_zero_init_mask_tokens (`bool`, *optional*, defaults to `True`):
        Initialize the mask tokens in the predictor with 0.
    pred_mlp_ratio (`float`, *optional*, defaults to 4.0):
        Ratio of the hidden size of the MLPs used in Predictor relative to the `pred_hidden_size`.

    Example:

    ```python
    >>> from vjepa21 import VJEPA21Config, VJEPA21Model

    >>> # Initializing a VJEPA21 style configuration
    >>> configuration = VJEPA21Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = VJEPA21Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vjepa2_1"

    patch_size: int | list[int] | tuple[int, int] = 16
    crop_size: int = 384
    frames_per_clip: int = 64
    tubelet_size: int = 2
    hidden_size: int = 1024
    in_chans: int = 3
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    drop_path_rate: float | int = 0.0
    mlp_ratio: int | float = 4.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    attention_probs_dropout_prob: float | int = 0.0
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0
    num_pooler_layers: int = 3
    pred_hidden_size: int = 384
    pred_num_attention_heads: int = 12
    pred_num_hidden_layers: int = 12
    pred_num_mask_tokens: int = 8
    pred_zero_init_mask_tokens: bool = True
    pred_mlp_ratio: int | float = 4.0
    img_temporal_dim_size: int = 1
    modality_embedding: bool = True
    n_output_distillation: int = 4
    interpolate_rope: bool = True
    pretrained_crop_size: int = 256
    teacher_embed_dim: int | None = None
    pred_has_context_proj: bool = True


__all__ = ["VJEPA21Config"]
