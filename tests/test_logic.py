"""Offline checks for training/inference logic that must match the paper recipe."""

from __future__ import annotations

import os
import sys
import tempfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from drm.utils import HEAD_LAYER_MISSING, REWARD_HEADS
from drm.utils import (
    find_latest_checkpoint,
    find_weight_file,
    is_accelerate_state,
    parse_checkpoint_step,
)


def test_head_layer_missing_matches_original():
    assert REWARD_HEADS == ("2BConvGN", "7BConvGN")
    assert HEAD_LAYER_MISSING["2BConvGN"] == 3
    assert HEAD_LAYER_MISSING["7BConvGN"] == 5


def test_early_exit_index():
    """layer_missing=k breaks after block index len-k-1, i.e. skip the last k blocks."""
    num_layers = 24  # SD3.5 Medium
    layer_missing = 3
    break_at = num_layers - layer_missing - 1
    skipped = list(range(break_at + 1, num_layers))
    assert skipped == [21, 22, 23]

    num_layers = 38  # SD3.5 Large
    layer_missing = 5
    break_at = num_layers - layer_missing - 1
    skipped = list(range(break_at + 1, num_layers))
    assert skipped == [33, 34, 35, 36, 37]


def test_checkpoint_name_parsing():
    assert parse_checkpoint_step("checkpoint-2000") == 2000
    assert parse_checkpoint_step("checkpoint-epoch-0-step-2000") is None
    assert parse_checkpoint_step("checkpoint-0") == 0

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "checkpoint-100"))
        os.makedirs(os.path.join(tmp, "checkpoint-epoch-0-step-5000"))
        os.makedirs(os.path.join(tmp, "checkpoint-2000"))
        open(os.path.join(tmp, "checkpoint-2000", "model.safetensors"), "w").close()
        latest = find_latest_checkpoint(tmp)
        assert latest.endswith("checkpoint-2000")
        assert find_weight_file(latest).endswith("model.safetensors")
        assert is_accelerate_state(latest) is False


def test_bt_and_flow_if_torch_available():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("skip torch checks (torch not installed)")
        return

    def bradley_terry_loss(rewards_win, rewards_lose):
        return -nn.functional.logsigmoid(rewards_win - rewards_lose).mean()

    win = torch.tensor([[2.0]])
    lose = torch.tensor([[0.0]])
    good = bradley_terry_loss(win, lose)
    bad = bradley_terry_loss(lose, win)
    assert good < bad
    assert torch.isclose(good, -nn.functional.logsigmoid(torch.tensor(2.0)), atol=1e-5)

    x0 = torch.ones(2, 16, 4, 4)
    noise = torch.zeros_like(x0)
    sigma = torch.tensor([0.25]).view(1, 1, 1, 1)
    noisy = (1.0 - sigma) * x0 + sigma * noise
    assert torch.allclose(noisy, torch.full_like(x0, 0.75))


if __name__ == "__main__":
    test_head_layer_missing_matches_original()
    test_early_exit_index()
    test_checkpoint_name_parsing()
    test_bt_and_flow_if_torch_available()
    print("all logic checks passed")
