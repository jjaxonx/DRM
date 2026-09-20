#!/usr/bin/env bash
# Pairwise accuracy. Edit checkpoint and data paths before running.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python eval/infer_pairwise.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --reward_head 2BConvGN \
  --reward_model_path outputs/drm-sd35-medium-2BConvGN/checkpoint-2000/model.safetensors \
  --dataset_path /path/to/HPDv3/test.json \
  --image_folder /path/to/HPDv3 \
  --size 512 \
  --output_file results/drm-eval-hpdv3.jsonl
