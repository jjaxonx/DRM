#!/usr/bin/env bash
# SD3.5 Large + 7BConvGN. Edit data paths before running.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

accelerate launch \
  --config_file accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 8 \
  drm/train.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-large \
  --reward_head 7BConvGN \
  --max_size 512 \
  --train_json /path/to/HPDv3/train.json \
  --image_folder /path/to/HPDv3 \
  --confidence_threshold 0.95 \
  --uniform \
  --shift \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --checkpointing_steps 2000 \
  --num_train_epochs 1 \
  --output_dir outputs/drm-sd35-large-7BConvGN
