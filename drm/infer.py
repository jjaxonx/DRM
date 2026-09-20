#!/usr/bin/env python
"""Score one image (or a prompt+image pair) with a trained DRM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from drm.pipeline import DRMPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run DRM inference.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to model.safetensors")
    parser.add_argument("--reward_head", type=str, default="2BConvGN", choices=["2BConvGN", "7BConvGN"])
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--time", type=int, default=0, help="0 means no extra noise (typical eval).")
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--max_sequence_length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    pipe = DRMPipeline.from_sd3(
        args.pretrained_model_name_or_path,
        reward_head=args.reward_head,
        image_size=args.image_size,
        max_sequence_length=args.max_sequence_length,
    )
    pipe.load_reward_model(args.reward_model_path, device=args.device, dtype=dtype)
    pipe.to(device=args.device, dtype=dtype)
    score = pipe(image=args.image, prompt=args.prompt, time=args.time, add_noise=args.add_noise)
    print(float(score.squeeze().item()))


if __name__ == "__main__":
    main()
