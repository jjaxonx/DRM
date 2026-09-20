"""Small helpers shared by training and tests."""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
WEIGHT_NAMES = ("model.safetensors", "model.pth", "model.pt")

REWARD_HEADS = ("2BConvGN", "7BConvGN")
HEAD_LAYER_MISSING = {
    "2BConvGN": 3,  # SD3.5 Medium: skip last 3 MMDiT blocks
    "7BConvGN": 5,  # SD3.5 Large: skip last 5 MMDiT blocks
}


def parse_checkpoint_step(name: str) -> Optional[int]:
    """Parse `checkpoint-1234`. Ignore `checkpoint-epoch-0-step-1234`."""
    match = CHECKPOINT_DIR_RE.match(name.rstrip("/"))
    return int(match.group(1)) if match else None


def list_step_checkpoints(output_dir: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(output_dir):
        return []
    found = []
    for name in os.listdir(output_dir):
        step = parse_checkpoint_step(name)
        if step is not None:
            found.append((step, os.path.join(output_dir, name)))
    found.sort(key=lambda item: item[0])
    return found


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    checkpoints = list_step_checkpoints(output_dir)
    return checkpoints[-1][1] if checkpoints else None


def is_accelerate_state(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return any(
        os.path.exists(os.path.join(path, name))
        for name in ("optimizer.bin", "scheduler.bin", "random_states_0.pkl")
    )


def find_weight_file(path: str) -> Optional[str]:
    if os.path.isfile(path) and path.endswith((".safetensors", ".pth", ".pt")):
        return path
    if not os.path.isdir(path):
        return None
    for name in WEIGHT_NAMES:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate
    return None
