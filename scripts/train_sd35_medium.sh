#!/usr/bin/env bash
# SD3.5 Medium + 2BConvGN. Edit data paths before running.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

accelerate launch \
  --config_file accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 8 \
  drm/train.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --reward_head 2BConvGN \
  --max_size 512 \
  --train_json /path/to/HPDv3/train.json \
  --image_folder /path/to/HPDv3 \
  --confidence_threshold 0.95 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --checkpointing_steps 2000 \
  --num_train_epochs 1 \
  --output_dir outputs/drm-sd35-medium-2BConvGN
