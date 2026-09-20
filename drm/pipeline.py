"""Inference pipeline: encode image + prompt, then score with a DRM reward head."""

from __future__ import annotations

from typing import List, Optional, Union

import PIL.Image
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from torchvision import transforms
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from drm.model import REWARD_HEADS, build_reward_model
from drm.transformer import SD3Transformer2DModel

logger = logging.get_logger(__name__)


class DRMPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, SD3IPAdapterMixin):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        image_encoder=None,
        feature_extractor=None,
        reward_head: str = "2BConvGN",
        image_size: int = 512,
        max_sequence_length: int = 512,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, vae_latent_channels=latent_channels
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 77
        )
        if reward_head not in REWARD_HEADS:
            raise ValueError(f"Unknown reward_head={reward_head!r}. Expected one of {REWARD_HEADS}.")
        self.reward_head = reward_head
        self.reward_model = build_reward_model(self.transformer, reward_head)
        self.image_size = image_size
        self.max_sequence_length = max_sequence_length

    @classmethod
    def from_sd3(
        cls,
        pretrained_model_name_or_path: str,
        reward_head: str = "2BConvGN",
        image_size: int = 512,
        max_sequence_length: int = 512,
        **kwargs,
    ):
        transformer = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer"
        )
        return cls.from_pretrained(
            pretrained_model_name_or_path,
            transformer=transformer,
            reward_head=reward_head,
            image_size=image_size,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )

    def load_reward_model(self, reward_model_path: str, device: str = "cuda", dtype=torch.float16):
        if reward_model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(reward_model_path, device="cpu")
        elif reward_model_path.endswith(".pt") or reward_model_path.endswith(".pth"):
            state_dict = torch.load(reward_model_path, map_location="cpu")
        else:
            raise ValueError(f"Expected .safetensors / .pt / .pth, got {reward_model_path}")
        self.reward_model.load_state_dict(state_dict, strict=True)
        self.reward_model.to(device=device, dtype=dtype)
        self.reward_model.eval()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if getattr(self, "reward_model", None) is not None:
            self.reward_model.to(*args, **kwargs)
        return self

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder_3(text_inputs.input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device
        tokenizer = [self.tokenizer, self.tokenizer_2][clip_model_index]
        text_encoder = [self.text_encoder, self.text_encoder_2][clip_model_index]
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2].to(dtype=self.text_encoder.dtype, device=device)
        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(self, prompt: Union[str, List[str]], max_sequence_length: Optional[int] = None, device=None):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        max_sequence_length = self.max_sequence_length if max_sequence_length is None else max_sequence_length
        prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(prompt, device=device, clip_model_index=0)
        prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(prompt, device=device, clip_model_index=1)
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        t5_prompt_embed = self._get_t5_prompt_embeds(
            prompt=prompt, max_sequence_length=max_sequence_length, device=device
        )
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device=self.device)
        timesteps = timesteps.to(device=self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _prepare_pixels(self, image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            pixel_values = image if image.ndim == 4 else image.unsqueeze(0)
            return pixel_values.to(device=self.device, dtype=self.vae.dtype)

        if isinstance(image, str):
            image = PIL.Image.open(image).convert("RGB")
        elif isinstance(image, PIL.Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError("image must be a path, PIL.Image, or Tensor")
        # Match training dataset: PIL resize (BICUBIC), then ToTensor + Normalize.
        image = image.resize((self.image_size, self.image_size), resample=PIL.Image.BICUBIC)
        pixel_values = transforms.ToTensor()(image)
        pixel_values = transforms.Normalize([0.5], [0.5])(pixel_values).unsqueeze(0)
        return pixel_values.to(device=self.device, dtype=self.vae.dtype)

    @torch.no_grad()
    def __call__(
        self,
        image,
        prompt: Union[str, List[str]] = "",
        time: int = 0,
        add_noise: bool = False,
    ) -> torch.Tensor:
        self.reward_model.eval()
        reward_dtype = next(self.reward_model.parameters()).dtype
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt, max_sequence_length=self.max_sequence_length
        )
        pixel_values = self._prepare_pixels(image)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latents = latents.to(dtype=reward_dtype)

        if time == 0:
            timesteps = torch.zeros([pixel_values.shape[0]], dtype=torch.float32, device=self.device)
        else:
            timesteps = (
                self.scheduler.timesteps[-time]
                .reshape(-1)
                .repeat(pixel_values.shape[0])
                .to(dtype=torch.float32, device=self.device)
            )
            if add_noise:
                noise = torch.randn_like(latents)
                sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                latents = (1.0 - sigmas) * latents + sigmas * noise

        return self.reward_model(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds.to(dtype=reward_dtype),
            pooled_projections=pooled_prompt_embeds.to(dtype=reward_dtype),
        )
