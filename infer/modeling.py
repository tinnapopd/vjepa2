# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from transformers import initialization as init  # type: ignore
from transformers.activations import ACT2FN  # type: ignore
from transformers.modeling_layers import GradientCheckpointingLayer  # type: ignore
from transformers.modeling_outputs import (  # type: ignore
    BaseModelOutput,
    ImageClassifierOutput,
)
from transformers.modeling_utils import (  # type: ignore
    ALL_ATTENTION_FUNCTIONS,
    PreTrainedModel,
)  # type: ignore
from transformers.processing_utils import Unpack  # type: ignore
from transformers.utils import (  # type: ignore
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from transformers.utils.generic import merge_with_config_defaults  # type: ignore
from transformers.utils.output_capturing import OutputRecorder, capture_outputs  # type: ignore
from .configuration import VJEPA21Config  # type: ignore


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    VJEPA 2.1 Predictor outputs that also contains the masked encoder outputs
    """
)
class VJEPA21WithMaskedInputPredictorOutput(ModelOutput):
    r"""
    masked_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, returned when `context_mask` is provided which is applied on VJEPA21Encoder outputs):
        The masked hidden state of the model.
    target_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, returned when `target_mask` is provided which is applied on VJEPA21Encoder outputs):
        The target hidden state of the model.
    """

    last_hidden_state: torch.FloatTensor
    masked_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    target_hidden_state: torch.FloatTensor | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    VJEPA 2.1 outputs that also contains the masked encoder outputs
    Optionally contains the predictor outputs
    """
)
class VJEPA21WithMaskedInputModelOutput(ModelOutput):
    r"""
    masked_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, returned when `context_mask` is provided which is applied on VJEPA21Encoder outputs):
        The masked hidden state of the model.
    predictor_output (`VJEPA21WithMaskedInputPredictorOutput`, *optional*):
        The output from the Predictor module.
    """

    last_hidden_state: torch.FloatTensor
    masked_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    predictor_output: VJEPA21WithMaskedInputPredictorOutput | None = None

    def to_tuple(self):
        output = list(super().to_tuple())
        if isinstance(output[-1], VJEPA21WithMaskedInputPredictorOutput):
            output[-1] = output[-1].to_tuple()
        return tuple(output)


class VJEPA21PatchEmbeddings3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        config: VJEPA21Config,
        hidden_size: int = 1024,
        tubelet_size: int | None = None,
    ):
        super().__init__()
        self.patch_size = config.patch_size
        self.tubelet_size = (
            tubelet_size if tubelet_size is not None else config.tubelet_size
        )
        self.hidden_size = hidden_size

        self.proj = nn.Conv3d(
            in_channels=config.in_chans,
            out_channels=hidden_size,
            kernel_size=(
                self.tubelet_size,
                config.patch_size,
                config.patch_size,
            ),
            stride=(self.tubelet_size, config.patch_size, config.patch_size),
        )

    @staticmethod
    def num_patches(config):
        return (
            (config.frames_per_clip // config.tubelet_size)
            * (config.crop_size // config.patch_size)
            * (config.crop_size // config.patch_size)
        )

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        x = self.proj(pixel_values_videos).flatten(2).transpose(1, 2)
        return x


class VJEPA21Embeddings(nn.Module):
    """
    Construct mask token, position and patch embeddings.
    Supports separate image and video patch embeddings with modality embeddings.
    """

    def __init__(self, config: VJEPA21Config, hidden_size: int = 1024):
        super().__init__()

        self.config = config
        self.hidden_size = hidden_size
        # Video patch embeddings (uses config tubelet_size)
        self.patch_embeddings = VJEPA21PatchEmbeddings3D(
            config, hidden_size=hidden_size
        )
        # Image patch embeddings (uses tubelet_size=1)
        self.patch_embeddings_img = VJEPA21PatchEmbeddings3D(
            config, hidden_size=hidden_size, tubelet_size=1
        )

        self.num_patches = self.patch_embeddings.num_patches
        self.patch_size = config.patch_size

        # Modality embeddings
        if config.modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(
        self, pixel_values_videos: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        num_frames = pixel_values_videos.shape[1]

        # Swap `frames` and `channels` dims, the result is:
        # (batch_size, channels, num_frames, height, width)
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4)

        # Detect if input is image based on temporal dimension
        is_image = (
            pixel_values_videos.shape[2] == self.config.img_temporal_dim_size
        )

        # For some cases, if the input vision (image/video) consists of num_frames < tubelet_size,
        # then embedding lookup fails. In these cases, we duplicate the frames.
        if is_image:
            target_dtype = self.patch_embeddings_img.proj.weight.dtype
            pixel_values_videos = pixel_values_videos.to(dtype=target_dtype)
            embeddings = self.patch_embeddings_img(pixel_values_videos)
        else:
            if num_frames < self.config.tubelet_size:
                pixel_values_videos = pixel_values_videos.repeat(
                    1, 1, self.config.tubelet_size, 1, 1
                )
            target_dtype = self.patch_embeddings.proj.weight.dtype
            pixel_values_videos = pixel_values_videos.to(dtype=target_dtype)
            embeddings = self.patch_embeddings(pixel_values_videos)

        # Add modality embeddings
        if self.config.modality_embedding:
            if is_image:
                embeddings = embeddings + self.img_mod_embed
            else:
                embeddings = embeddings + self.video_mod_embed

        return embeddings, is_image


# Adapted from transformers.models.vit.modeling_vit.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()

    # similar to inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    # they are computing this every time. instead HF style is to compute the inv_freq once and store it
    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = pos.unsqueeze(-1) * omega  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)

    emb_sin = emb_sin.repeat_interleave(2, dim=-1)
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)

    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return ((x * emb_cos) + (y * emb_sin)).to(x.dtype)


class VJEPA21RopeAttention(nn.Module):
    def __init__(
        self,
        config: VJEPA21Config,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {(hidden_size,)} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = (
            self.num_attention_heads * self.attention_head_size
        )

        self.query = nn.Linear(
            hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            hidden_size, self.all_head_size, bias=config.qkv_bias
        )

        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

        self.grid_size = self.config.crop_size // self.config.patch_size
        self.grid_depth = (
            self.config.frames_per_clip // self.config.tubelet_size
        )
        self.pretrained_grid_size = (
            self.config.pretrained_crop_size // self.config.patch_size
        )

        self.d_dim = 2 * ((self.attention_head_size // 3) // 2)
        self.h_dim = 2 * ((self.attention_head_size // 3) // 2)
        self.w_dim = 2 * ((self.attention_head_size // 3) // 2)

        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False
        self.interpolate_rope = config.interpolate_rope

    def _get_frame_pos(self, ids):
        tokens_per_frame = int(self.grid_size * self.grid_size)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids):
        # Remove frame component from ids
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        # --
        tokens_per_row = self.grid_size
        return ids // tokens_per_row

    def get_position_ids(self, x, masks=None):
        device = x.device
        token_size = x.size(1)

        # Note: when masks is none, we use a 1d id instead of Bxnum_attention_heads mask,
        # as 1d vector is broadcasted to the correct shapes.
        if masks is not None:
            ids = masks.unsqueeze(1).repeat(1, self.num_attention_heads, 1)
        else:
            ids = torch.arange(token_size, device=device)

        # Convert to float when interpolate_rope is True (for fractional positions)
        if self.interpolate_rope:
            ids = ids.float()

        # change to allow for extrapolation
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        # --
        tokens_per_row = self.grid_size
        height_ids = self._get_height_pos(ids)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (
            ids - tokens_per_frame * frame_ids
        ) - tokens_per_row * height_ids

        # Scale height/width positions to pretrained grid size range for RoPE interpolation
        if self.interpolate_rope:
            height_ids = (
                height_ids
                * (self.pretrained_grid_size - 1)
                / (self.grid_size - 1)
            )
            width_ids = (
                width_ids
                * (self.pretrained_grid_size - 1)
                / (self.grid_size - 1)
            )

        return frame_ids, height_ids, width_ids

    def apply_rotary_embeddings(self, qk, pos_ids):
        d_mask, h_mask, w_mask = pos_ids
        s = 0
        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim
        # Combine rotated dimension
        if s < self.attention_head_size:
            qkr = qk[..., s:]
            qk = torch.cat([qkd, qkh, qkw, qkr], dim=-1)
        else:
            qk = torch.cat([qkd, qkh, qkw], dim=-1)
        return qk

    def forward(
        self,
        hidden_states,
        position_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)
        query_layer = (
            self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = (
            self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        )

        pos_ids = self.get_position_ids(hidden_states, masks=position_mask)
        key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)
        query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            None,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )
        context_layer = self.proj(
            context_layer.reshape(new_context_layer_shape)
        )

        return context_layer, attention_probs


# Adapted from transformers.models.beit.modeling_dinov2.drop_path
def drop_path(
    input: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (
        input.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=input.dtype, device=input.device
    )
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Adapted from transformers.models.beit.modeling_beit.BeitDropPath
class VJEPA21DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float | None = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)  # type: ignore

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class VJEPA21MLP(nn.Module):
    def __init__(
        self,
        config: VJEPA21Config,
        hidden_size: int = 1024,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        in_features = out_features = hidden_size
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.activation = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class VJEPA21Layer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(
        self,
        config: VJEPA21Config,
        drop_path_rate: float = 0.0,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = VJEPA21RopeAttention(
            config, hidden_size, num_attention_heads
        )
        self.drop_path = (
            VJEPA21DropPath(drop_path_rate)
            if config.drop_path_rate > 0.0
            else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA21MLP(
            config, hidden_size=hidden_size, mlp_ratio=mlp_ratio
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, ...]:
        # Self-Attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attention_output, attn_weights = self.attention(
            hidden_states,
            position_mask=position_mask,  # position mask for context/target selection
        )
        hidden_states = self.drop_path(attention_output) + residual

        # MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        # Add self attentions if we output attention weights
        return hidden_states, attn_weights


def apply_masks(
    tensor: torch.Tensor, masks: list[torch.Tensor]
) -> torch.Tensor:
    """
    Args:
        tensor (`torch.Tensor`):
            Tensor of shape [batch_size, num_patches, feature_dim]
        masks (`List[torch.Tensor]`):
            List of tensors of shape [batch_size, num_patches] containing indices of patches to keep
    """
    all_masked_tensors = []
    for mask in masks:
        mask = mask.to(tensor.device)
        mask_keep = mask.unsqueeze(-1).repeat(1, 1, tensor.size(-1))
        all_masked_tensors += [torch.gather(tensor, dim=1, index=mask_keep)]

    return torch.cat(all_masked_tensors, dim=0)


def _get_hierarchical_layers(depth: int) -> list[int]:
    """Return layer indices for hierarchical output extraction."""
    if depth == 12:
        return [2, 5, 8, 11]
    elif depth == 24:
        return [5, 11, 17, 23]
    elif depth == 40:
        return [9, 19, 29, 39]
    elif depth == 48:
        return [11, 23, 37, 47]
    else:
        # Default: evenly spaced 4 layers ending at last layer
        n = 4
        step = depth // n
        return [step * (i + 1) - 1 for i in range(n)]


class VJEPA21Encoder(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config

        self.embeddings = VJEPA21Embeddings(
            config, hidden_size=config.hidden_size
        )
        drop_path_rates = [
            (
                config.drop_path_rate * i / (config.num_hidden_layers - 1)
                if config.num_hidden_layers > 1
                else 0.0
            )
            for i in range(config.num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [
                VJEPA21Layer(
                    config,
                    drop_path_rate=drop_path_rates[i],
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Deep self-supervision: norms_block instead of single layernorm
        self.hierarchical_layers = _get_hierarchical_layers(
            config.num_hidden_layers
        )
        self.norms_block = nn.ModuleList(
            [
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                for _ in self.hierarchical_layers
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        pixel_values_videos: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[BaseModelOutput, bool]:
        embeddings, is_image = self.embeddings(pixel_values_videos)
        hidden_states = embeddings

        # Store intermediate outputs for hierarchical layers
        hierarchical_outputs = []
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, None, **kwargs)
            hidden_states = layer_outputs[0]
            if i in self.hierarchical_layers:
                hierarchical_outputs.append(hidden_states)

        # At inference, apply the last norm block
        hidden_states = self.norms_block[-1](hidden_states)

        return (  # type: ignore
            BaseModelOutput(
                last_hidden_state=hidden_states,
            ),
            is_image,
            hierarchical_outputs,
        )


# ============================================================================
# Predictor
# ============================================================================


class VJEPA21PredictorEmbeddings(nn.Module):
    """
    Construct mask token, position and patch embeddings for the predictor.
    Supports hierarchical layer inputs via concatenation and projection.
    """

    def __init__(self, config: VJEPA21Config):
        super().__init__()

        self.config = config
        n_distill = config.n_output_distillation

        # Predictor embeddings: hierarchical concatenation -> projection
        if n_distill > 1:
            self.predictor_embeddings = nn.Sequential(
                nn.Linear(config.hidden_size * n_distill, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.pred_hidden_size),
            )
        else:
            self.predictor_embeddings = nn.Linear(  # type: ignore
                config.hidden_size, config.pred_hidden_size
            )

        self.num_mask_tokens = 0
        self.zero_init_mask_tokens = config.pred_zero_init_mask_tokens
        self.num_mask_tokens = config.pred_num_mask_tokens
        self.mask_tokens = nn.Parameter(
            torch.zeros(self.num_mask_tokens, 1, 1, config.pred_hidden_size)
        )

        self.patch_size = config.patch_size
        self.config = config

        # Modality embeddings
        if config.modality_embedding:
            self.img_mod_embed = nn.Parameter(
                torch.zeros(1, 1, config.pred_hidden_size)
            )
            self.video_mod_embed = nn.Parameter(
                torch.zeros(1, 1, config.pred_hidden_size)
            )

    @staticmethod
    def num_patches(config):
        if config.frames_per_clip > 1:
            return (
                (config.frames_per_clip // config.tubelet_size)
                * (config.crop_size // config.patch_size)
                * (config.crop_size // config.patch_size)
            )
        else:
            return (config.crop_size // config.patch_size) * (
                config.crop_size // config.patch_size
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        mask_index: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states : encoder outputs (context), possibly concatenated hierarchical outputs
        context_mask: tokens of the context (outputs from the encoder)
        target_mask: tokens to predict
        mask_index: index of the target mask to choose
        """

        B = hidden_states.size(0)
        context = self.predictor_embeddings(hidden_states)

        # Make target tokens
        mask_index = mask_index % self.num_mask_tokens
        target = self.mask_tokens[mask_index]

        # Use the provided target mask to get the max patch num
        max_patch_num = (
            target_mask[0].max() + 1
        )  # one extra to include the last patch
        target = target.repeat(B, max_patch_num, 1)  # type: ignore
        target = apply_masks(target, target_mask)

        # Concatenate context & target tokens
        context = context.repeat(len(context_mask), 1, 1)
        embeddings = torch.cat([context, target], dim=1)

        # Positions of context & target tokens
        cm = torch.cat(context_mask, dim=0)
        tm = torch.cat(target_mask, dim=0)
        masks = torch.cat([cm, tm], dim=1)

        return embeddings, masks


class VJEPA21Predictor(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embeddings = VJEPA21PredictorEmbeddings(config)

        drop_path_rates = [
            (
                config.drop_path_rate * i / (config.pred_num_hidden_layers - 1)
                if config.pred_num_hidden_layers > 1
                else 0.0
            )
            for i in range(config.pred_num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [
                VJEPA21Layer(
                    config,
                    drop_path_rate=drop_path_rates[i],
                    hidden_size=config.pred_hidden_size,
                    num_attention_heads=config.pred_num_attention_heads,
                    mlp_ratio=config.pred_mlp_ratio,
                )
                for i in range(config.pred_num_hidden_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(
            config.pred_hidden_size, eps=config.layer_norm_eps
        )

        # Output projection
        hierarchical_layers = _get_hierarchical_layers(
            config.num_hidden_layers
        )
        n_hier = min(config.n_output_distillation, len(hierarchical_layers))
        if config.teacher_embed_dim is not None:
            out_dim = config.teacher_embed_dim // n_hier
        else:
            out_dim = config.hidden_size
        self.proj = nn.Linear(
            config.pred_hidden_size, n_hier * out_dim, bias=True
        )

        # Context projection
        if config.pred_has_context_proj:
            self.proj_context = nn.Linear(
                config.pred_hidden_size, n_hier * out_dim, bias=True
            )
        else:
            self.proj_context = None  # type: ignore

        # Modality embeddings
        if config.modality_embedding:
            self.img_mod_embed = nn.Parameter(
                torch.zeros(1, 1, config.pred_hidden_size)
            )
            self.video_mod_embed = nn.Parameter(
                torch.zeros(1, 1, config.pred_hidden_size)
            )

    def sort_tokens(self, hidden_states, position_masks, argsort):
        # gather position masks
        argsort = argsort.to(position_masks.device)
        position_masks = torch.gather(position_masks, dim=1, index=argsort)

        # gather hidden states
        argsort = argsort.to(hidden_states.device)
        hidden_states_argsort = argsort.unsqueeze(-1).expand(
            -1, -1, hidden_states.size(-1)
        )
        hidden_states = torch.gather(
            hidden_states, dim=1, index=hidden_states_argsort
        )

        return hidden_states, position_masks

    def unsort_tokens(self, hidden_states, argsort):
        argsort = argsort.to(hidden_states.device)
        reverse_argsort = torch.argsort(argsort, dim=1)
        reverse_argsort = reverse_argsort.unsqueeze(-1).expand(
            -1, -1, hidden_states.size(-1)
        )
        hidden_states = torch.gather(
            hidden_states, dim=1, index=reverse_argsort
        )
        return hidden_states

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        is_image: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        # mask out the encoder hidden states
        encoder_hidden_states = apply_masks(
            encoder_hidden_states, context_mask
        )
        _, N_ctxt, D = encoder_hidden_states.shape
        hidden_states, position_masks = self.embeddings(
            encoder_hidden_states, context_mask, target_mask
        )

        # Put tokens in sorted order
        argsort = torch.argsort(position_masks, dim=1)  # [B, N]
        hidden_states, position_masks = self.sort_tokens(
            hidden_states, position_masks, argsort
        )

        # Apply modality embedding AFTER sorting tokens, BEFORE transformer layers
        if self.config.modality_embedding:
            if is_image:
                hidden_states = hidden_states + self.img_mod_embed
            else:
                hidden_states = hidden_states + self.video_mod_embed

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states, position_masks, **kwargs
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.layernorm(hidden_states)
        # unsort and extract the predicted tokens
        hidden_states = self.unsort_tokens(hidden_states, argsort)

        # Split context and target tokens
        context_hidden = hidden_states[:, :N_ctxt]
        target_hidden = hidden_states[:, N_ctxt:]

        # Apply projections
        target_hidden = self.proj(target_hidden)
        if self.proj_context is not None:
            context_hidden = self.proj_context(context_hidden)

        # Return target predictions as last_hidden_state
        hidden_states = target_hidden

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


# ============================================================================
# Pooler classes
# ============================================================================


class VJEPA21PoolerSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Input shape: Batch x Time x Channel"""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        queries = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class VJEPA21PoolerCrossAttention(nn.Module):
    """It's different from other cross-attention layers, doesn't have output projection layer (o_proj)"""

    # in case of modular refactoring - o_proj can be replaces with nn.Identity()

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_seq_length, embed_dim = queries.shape
        kv_seq_length = keys.shape[1]

        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        queries = queries.view(
            batch_size, q_seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, kv_seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, kv_seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(
            batch_size, q_seq_length, embed_dim
        ).contiguous()

        return attn_output, attn_weights


# Modified from SiglipEncoderLayer, but we have to propagate proper hidden_size to VJEPA21MLP
class VJEPA21PoolerSelfAttentionLayer(GradientCheckpointingLayer):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.self_attn = VJEPA21PoolerSelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = VJEPA21MLP(config, hidden_size=config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights


class VJEPA21PoolerCrossAttentionLayer(GradientCheckpointingLayer):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.cross_attn = VJEPA21PoolerCrossAttention(config)
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = VJEPA21MLP(config, hidden_size=config.hidden_size)

    def forward(
        self,
        queries: torch.Tensor,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply cross-attention
        residual = queries
        hidden_state = self.layer_norm1(hidden_state)
        hidden_state, *attn_weights = self.cross_attn(
            queries,
            hidden_state,
            hidden_state,
            attention_mask=attention_mask,
        )
        hidden_state = residual + hidden_state

        # Apply MLP
        residual = hidden_state
        hidden_state = self.layer_norm2(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        return hidden_state, *attn_weights


class VJEPA21AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cross_attention_layer = VJEPA21PoolerCrossAttentionLayer(config)
        self.self_attention_layers = nn.ModuleList(
            [
                VJEPA21PoolerSelfAttentionLayer(config)
                for _ in range(config.num_pooler_layers)
            ]
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state, attention_mask=None)[0]
        queries = self.query_tokens.repeat(hidden_state.shape[0], 1, 1)
        hidden_state = self.cross_attention_layer(queries, hidden_state)[0]
        return hidden_state.squeeze(1)


# ============================================================================
# Top-level models
# ============================================================================


@auto_docstring
class VJEPA21PreTrainedModel(PreTrainedModel):
    config: VJEPA21Config
    base_model_prefix = "vjepa2_1"
    main_input_name = "pixel_values_videos"
    input_modalities = "video"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "VJEPA21Layer",
        "VJEPA21PoolerSelfAttentionLayer",
        "VJEPA21PoolerCrossAttentionLayer",
        "VJEPA21PredictorEmbeddings",
        "VJEPA21Predictor",
        "VJEPA21Embeddings",
        "VJEPA21AttentivePooler",
    ]
    _supports_sdpa = True
    _supports_flash_attn = True
    _can_record_outputs = {
        "hidden_states": OutputRecorder(
            VJEPA21Layer, layer_name="encoder.layer"
        ),
        "attentions": OutputRecorder(
            VJEPA21RopeAttention, index=1, layer_name="encoder.layer"
        ),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""

        init_std = self.config.initializer_range
        if isinstance(module, VJEPA21AttentivePooler):
            init.trunc_normal_(module.query_tokens, std=init_std)
            for i, layer in enumerate(module.self_attention_layers, 1):
                std = init_std / (i**0.5)
                init.trunc_normal_(layer.self_attn.out_proj.weight, std=std)  # type: ignore
                init.trunc_normal_(layer.mlp.fc2.weight, std=std)  # type: ignore
            std = init_std / (len(module.self_attention_layers) + 1) ** 0.5
            init.trunc_normal_(
                module.cross_attention_layer.mlp.fc2.weight, std=std
            )
        elif isinstance(module, VJEPA21PredictorEmbeddings):
            if module.zero_init_mask_tokens:
                init.zeros_(module.mask_tokens)
            else:
                init.trunc_normal_(module.mask_tokens, std=init_std)
            if hasattr(module, "img_mod_embed"):
                init.zeros_(module.img_mod_embed)
            if hasattr(module, "video_mod_embed"):
                init.zeros_(module.video_mod_embed)
        elif isinstance(module, (VJEPA21Embeddings, VJEPA21Predictor)):
            if hasattr(module, "img_mod_embed"):
                init.zeros_(module.img_mod_embed)
            if hasattr(module, "video_mod_embed"):
                init.zeros_(module.video_mod_embed)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.trunc_normal_(module.weight, std=init_std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)


@auto_docstring
class VJEPA21Model(VJEPA21PreTrainedModel):
    def __init__(self, config: VJEPA21Config):
        super().__init__(config)
        self.config = config

        self.encoder = VJEPA21Encoder(config)
        self.predictor = VJEPA21Predictor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> VJEPA21PatchEmbeddings3D:
        return self.encoder.embeddings.patch_embeddings

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        context_mask: list[torch.Tensor] | None = None,
        target_mask: list[torch.Tensor] | None = None,
        skip_predictor: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> VJEPA21WithMaskedInputModelOutput:
        r"""
        context_mask (`torch.Tensor` with shape `[batch_size, patch_size, 1]`, *optional*):
            The mask position ids indicating which encoder output patches are going to be exposed to the predictor.
            By default, this mask is created as torch.arange(N).unsqueeze(0).repeat(B,1), indicating full context
            available to the predictor.
        target_mask (`torch.Tensor` with shape `[batch_size, patch_size, 1]`, *optional*):
            The mask position ids indicating which encoder output patches are going to be used as a prediction target
            for the predictor. By default, this mask is created as torch.arange(N).unsqueeze(0).repeat(B,1), indicating
            that the predictor should predict all encoder patches.
        skip_predictor (bool):
            flag to skip the predictor forward, useful if you just need the encoder outputs
        """
        if pixel_values_videos is None:
            raise ValueError("You have to specify pixel_values_videos")

        encoder_result = self.encoder(
            pixel_values_videos=pixel_values_videos,
            **kwargs,
        )
        encoder_outputs, is_image, hierarchical_outputs = encoder_result
        sequence_output = encoder_outputs.last_hidden_state

        if context_mask is None and target_mask is None:
            B = pixel_values_videos.size(0)
            N = sequence_output.size(
                1
            )  # ensure we are using dynamic patch size
            context_mask = [
                torch.arange(N, device=pixel_values_videos.device)
                .unsqueeze(0)
                .repeat((B, 1))
            ]
            target_mask = [
                torch.arange(N, device=pixel_values_videos.device)
                .unsqueeze(0)
                .repeat((B, 1))
            ]

        if not skip_predictor:
            # Prepare hierarchical encoder outputs for predictor
            n_distill = self.config.n_output_distillation
            if n_distill > 1 and len(hierarchical_outputs) > 0:
                # Normalize each hierarchical output with its corresponding norm
                normed_outputs = []
                for idx, h_out in enumerate(hierarchical_outputs):
                    normed_outputs.append(self.encoder.norms_block[idx](h_out))
                # Apply context mask to each and concatenate along feature dim
                masked_hier = [
                    apply_masks(h, context_mask)  # type: ignore
                    for h in normed_outputs
                ]
                predictor_input = torch.cat(masked_hier, dim=-1)
            else:
                predictor_input = sequence_output

            predictor_outputs: BaseModelOutput = self.predictor(
                encoder_hidden_states=predictor_input
                if n_distill > 1 and len(hierarchical_outputs) > 0
                else sequence_output,
                context_mask=context_mask,
                target_mask=target_mask,
                is_image=is_image,
                **kwargs,
            )
            predictor_output = VJEPA21WithMaskedInputPredictorOutput(
                last_hidden_state=predictor_outputs.last_hidden_state,
                target_hidden_state=apply_masks(sequence_output, target_mask),  # type: ignore
                hidden_states=predictor_outputs.hidden_states,
                attentions=predictor_outputs.attentions,
            )
        else:
            predictor_output = None

        encoder_output = VJEPA21WithMaskedInputModelOutput(
            last_hidden_state=sequence_output,
            masked_hidden_state=apply_masks(sequence_output, context_mask),  # type: ignore
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            predictor_output=predictor_output,
        )

        return encoder_output

    def get_vision_features(self, pixel_values_videos) -> torch.Tensor:
        encoder_output = self.forward(pixel_values_videos, skip_predictor=True)
        return encoder_output.last_hidden_state


@auto_docstring(
    custom_intro="""
    V-JEPA 2.1 Model transformer with a video classification head on top (a linear layer on top of the attentive pooler).
    """
)
class VJEPA21ForVideoClassification(VJEPA21PreTrainedModel):
    def __init__(self, config: VJEPA21Config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vjepa2_1 = VJEPA21Model(config)

        # Classifier head
        self.pooler = VJEPA21AttentivePooler(config)
        self.classifier = nn.Linear(
            config.hidden_size, config.num_labels, bias=True
        )

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.vjepa2_1(
            pixel_values_videos=pixel_values_videos,
            skip_predictor=True,
            **kwargs,
        )

        last_hidden_state = outputs.last_hidden_state
        pooler_output = self.pooler(last_hidden_state)
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                pooled_logits=logits, labels=labels, config=self.config
            )

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "VJEPA21Model",
    "VJEPA21PreTrainedModel",
    "VJEPA21ForVideoClassification",
]
