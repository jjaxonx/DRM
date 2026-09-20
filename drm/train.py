#!/usr/bin/env python
"""Train DRM with pairwise Bradley-Terry loss and mix_time noise.

Official recipe: freeze VAE + text encoders, train MMDiT + ConvGN head,
sample a shared flow-matching timestep per pair, add independent noise,
optimize -log sigmoid(s_win - s_lose).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, free_memory
from diffusers.utils import is_wandb_available
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from drm.dataset import PairwiseRewardDataset
from drm.model import REWARD_HEADS, build_reward_model
from drm.transformer import SD3Transformer2DModel
from drm.utils import (
    find_latest_checkpoint,
    find_weight_file,
    is_accelerate_state,
    list_step_checkpoints,
    parse_checkpoint_step,
)

if is_wandb_available():
    import wandb  # noqa: F401

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    if model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train a diffusion-based reward model (DRM).")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="SD3.5 Medium or Large checkpoint (HF id or local path).",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="outputs/drm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory, or 'latest'.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument(
        "--reward_head",
        type=str,
        default="2BConvGN",
        choices=list(REWARD_HEADS),
        help="2BConvGN for SD3.5 Medium, 7BConvGN for SD3.5 Large.",
    )
    parser.add_argument("--max_size", type=int, default=512, help="Square resize for training images.")
    parser.add_argument(
        "--train_json",
        type=str,
        nargs="+",
        required=True,
        help="Pairwise JSON files. path1 is the preferred image.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        nargs="+",
        default=None,
        help="Image root for each JSON. Use one path per JSON, or omit if paths are absolute.",
    )
    parser.add_argument("--confidence_threshold", type=float, default=0.95)
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="Sample mix_time timesteps uniformly instead of logit-normal.",
    )
    parser.add_argument(
        "--shift",
        action="store_true",
        help="Load the flow-matching scheduler with shift=1.0.",
    )
    parser.add_argument(
        "--save_full_state",
        action="store_true",
        help="Also write accelerate full state (optimizer) for resume. Always writes model.safetensors.",
    )

    args = parser.parse_args(input_args)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.image_folder is not None and len(args.image_folder) != len(args.train_json):
        raise ValueError("--image_folder must have the same number of entries as --train_json")
    return args


def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length, prompt, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    return prompt_embeds.to(dtype=text_encoder.dtype, device=device)


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    pooled = prompt_embeds[0]
    hidden = prompt_embeds.hidden_states[-2].to(dtype=text_encoder.dtype, device=device)
    return hidden, pooled


def encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    clip_embeds = []
    clip_pooled = []
    for tokenizer, text_encoder in zip(tokenizers[:2], text_encoders[:2]):
        hidden, pooled = _encode_prompt_with_clip(
            text_encoder, tokenizer, prompt, device=device or text_encoder.device
        )
        clip_embeds.append(hidden)
        clip_pooled.append(pooled)
    clip_prompt_embeds = torch.cat(clip_embeds, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled, dim=-1)
    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt,
        device=device or text_encoders[-1].device,
    )
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    return prompt_embeds, pooled_prompt_embeds


def bradley_terry_loss(rewards_win, rewards_lose):
    loss = -nn.functional.logsigmoid(rewards_win - rewards_lose)
    return loss.mean()


def load_reward_weights(model, weight_path: str):
    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(weight_path, device="cpu")
    else:
        state_dict = torch.load(weight_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    return missing, unexpected


def save_model_weights(accelerator, model, save_dir: str):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    # ZeRO-2 keeps parameters replicated; get_state_dict also works under DDP.
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        save_path = os.path.join(save_dir, "model.safetensors")
        cpu_state = {k: v.detach().contiguous().cpu() for k, v in state_dict.items()}
        save_file(cpu_state, save_path)
        logger.info(f"Saved inference weights to {save_path}")
    accelerator.wait_for_everyone()


def maybe_rotate_checkpoints(output_dir: str, checkpoints_total_limit: int):
    if checkpoints_total_limit is None:
        return
    checkpoints = list_step_checkpoints(output_dir)
    if len(checkpoints) < checkpoints_total_limit:
        return
    num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
    for _, path in checkpoints[:num_to_remove]:
        shutil.rmtree(path)
        logger.info(f"Removed old checkpoint {path}")


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir))
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision)
    tokenizer_three = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision)

    scheduler_kwargs = {"shift": 1.0} if args.shift else {}
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", **scheduler_kwargs
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = text_encoder_cls_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    model = build_reward_model(transformer, args.reward_head)

    model.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        model.transformer.enable_gradient_checkpointing()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = PairwiseRewardDataset(
        json_list=args.train_json,
        image_folder=args.image_folder,
        confidence_threshold=args.confidence_threshold,
        image_size=args.max_size,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=PairwiseRewardDataset.collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    def compute_text_embeddings(prompt):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
        return prompt_embeds.to(accelerator.device), pooled_prompt_embeds.to(accelerator.device)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("drm", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running DRM training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Reward head = {args.reward_head}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0
    last_saved_step = -1

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            path = find_latest_checkpoint(args.output_dir)
        else:
            path = args.resume_from_checkpoint
        if path is None:
            logger.info("No checkpoint found, starting a new run.")
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if is_accelerate_state(path):
                accelerator.load_state(path)
            else:
                weight_file = find_weight_file(path)
                if weight_file is None:
                    raise FileNotFoundError(
                        f"No Accelerate state or model weights in {path}. "
                        "Pass --save_full_state during training to resume optimizer state."
                    )
                load_reward_weights(accelerator.unwrap_model(model), weight_file)
                logger.info(
                    f"Loaded weights from {weight_file}. Optimizer/scheduler state was not restored."
                )
            step = parse_checkpoint_step(os.path.basename(path.rstrip("/")))
            if step is not None:
                global_step = step
                initial_global_step = global_step
                first_epoch = global_step // max(num_update_steps_per_epoch, 1)
                last_saved_step = global_step

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                model_dtype = next(model.parameters()).dtype
                pixel_values_1 = batch["pixel_values_1"].to(dtype=vae.dtype)
                pixel_values_2 = batch["pixel_values_2"].to(dtype=vae.dtype)
                prompts = batch["text_1"]

                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(prompts)
                    num_train_timesteps = int(noise_scheduler.config.num_train_timesteps)
                    if args.uniform:
                        indices = torch.randint(0, num_train_timesteps, (pixel_values_1.shape[0],)).long().cpu()
                    else:
                        u = compute_density_for_timestep_sampling(
                            weighting_scheme="logit_normal",
                            batch_size=pixel_values_1.shape[0],
                            logit_mean=0.0,
                            logit_std=1.0,
                            mode_scale=1.29,
                        )
                        indices = (u * num_train_timesteps).long().cpu()
                    # Keep timestep lookup in float32 so scheduler values match exactly.
                    timesteps = noise_scheduler.timesteps[indices].to(
                        device=accelerator.device, dtype=torch.float32
                    )

                    latents_win = vae.encode(pixel_values_1).latent_dist.sample()
                    latents_win = (latents_win - vae.config.shift_factor) * vae.config.scaling_factor
                    latents_win = latents_win.to(dtype=model_dtype)
                    noise_win = torch.randn_like(latents_win)
                    sigmas = get_sigmas(timesteps, n_dim=latents_win.ndim, dtype=latents_win.dtype)
                    latents_win = (1.0 - sigmas) * latents_win + sigmas * noise_win

                pred_win = model(
                    hidden_states=latents_win,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds.to(dtype=model_dtype),
                    pooled_projections=pooled_prompt_embeds.to(dtype=model_dtype),
                )

                with torch.no_grad():
                    latents_lose = vae.encode(pixel_values_2).latent_dist.sample()
                    latents_lose = (latents_lose - vae.config.shift_factor) * vae.config.scaling_factor
                    latents_lose = latents_lose.to(dtype=model_dtype)
                    noise_lose = torch.randn_like(latents_lose)
                    sigmas = get_sigmas(timesteps, n_dim=latents_lose.ndim, dtype=latents_lose.dtype)
                    latents_lose = (1.0 - sigmas) * latents_lose + sigmas * noise_lose

                pred_lose = model(
                    hidden_states=latents_lose,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds.to(dtype=model_dtype),
                    pooled_projections=pooled_prompt_embeds.to(dtype=model_dtype),
                )

                loss = bradley_terry_loss(pred_win, pred_lose)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    maybe_rotate_checkpoints(args.output_dir, args.checkpoints_total_limit)
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if args.save_full_state:
                        accelerator.save_state(save_dir)
                    save_model_weights(accelerator, model, save_dir)
                    last_saved_step = global_step

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break

        if global_step > last_saved_step:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            if args.save_full_state:
                accelerator.save_state(save_dir)
            save_model_weights(accelerator, model, save_dir)
            last_saved_step = global_step

    accelerator.wait_for_everyone()
    accelerator.end_training()
    free_memory()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    main(parse_args())
