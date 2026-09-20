"""DRM reward heads on top of SD3.5 MMDiT.

Official variants:
  - 2BConvGN: SD3.5 Medium, skip last 3 transformer blocks, Conv + GroupNorm head
  - 7BConvGN: SD3.5 Large, skip last 5 transformer blocks, Conv + GroupNorm head
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin, SD3Transformer2DLoadersMixin
from diffusers.models.modeling_utils import ModelMixin

from drm.transformer import SD3Transformer2DModel
from drm.utils import HEAD_LAYER_MISSING, REWARD_HEADS


class ConvScalarHeadGroupNorm(nn.Module):
    def __init__(self, in_channels: int, base_ch: int = 256, num_groups: int = 32):
        super().__init__()
        if base_ch % num_groups != 0 or (base_ch * 2) % num_groups != 0:
            raise ValueError(f"base_ch ({base_ch}) must be divisible by num_groups ({num_groups})")

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups, base_ch),
            nn.ReLU(inplace=False),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups, base_ch * 2),
            nn.ReLU(inplace=False),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups, base_ch * 2),
            nn.ReLU(inplace=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(base_ch * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.mlp(x)


class DRMRewardModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, SD3Transformer2DLoadersMixin):
    """MMDiT backbone + ConvGN scalar head. `layer_missing` selects which blocks to skip."""

    def __init__(self, transformer: SD3Transformer2DModel, layer_missing: int):
        super().__init__()
        self.transformer = transformer
        self.layer_missing = layer_missing
        self.reward_output = ConvScalarHeadGroupNorm(in_channels=self.transformer.out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
    ) -> torch.Tensor:
        _, features = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            joint_attention_kwargs=None,
            return_dict=False,
            layer_missing=self.layer_missing,
            inner_norm_proj=True,
        )
        return self.reward_output(features)


def build_reward_model(transformer: SD3Transformer2DModel, reward_head: str) -> DRMRewardModel:
    if reward_head not in HEAD_LAYER_MISSING:
        raise ValueError(f"Unknown reward_head={reward_head!r}. Expected one of {REWARD_HEADS}.")
    return DRMRewardModel(transformer=transformer, layer_missing=HEAD_LAYER_MISSING[reward_head])
