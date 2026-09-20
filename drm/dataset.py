"""Pairwise preference dataset. `path1` is always the preferred image."""

from __future__ import annotations

import json
import os
import random
from typing import List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class PairwiseRewardDataset(Dataset):
    """HPDv3-style pairwise JSON.

    Each record must contain `prompt`, `path1` (winner), `path2` (loser).
    Optional `confidence` is used for filtering.
    Images are resized to a square `image_size`.
    """

    def __init__(
        self,
        json_list: Sequence[str],
        image_folder: Optional[Sequence[Optional[str]]] = None,
        confidence_threshold: Optional[float] = 0.95,
        image_size: int = 512,
    ):
        self.samples = []
        for i, json_file in enumerate(json_list):
            with open(json_file, "r") as f:
                data = json.load(f)
            self.samples.extend((row, i) for row in data)

        if image_folder is None:
            self.image_folder = [None] * len(json_list)
        else:
            if len(image_folder) != len(json_list):
                raise ValueError("image_folder must have the same length as json_list")
            self.image_folder = list(image_folder)

        if confidence_threshold is not None:
            kept = []
            for sample, src in tqdm(self.samples, desc="Filtering by confidence"):
                conf = sample.get("confidence")
                if conf is None or conf >= confidence_threshold:
                    kept.append((sample, src))
            self.samples = kept

        self.image_size = image_size
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, sample: dict, key: str, src: int) -> str:
        path = sample[key]
        folder = self.image_folder[src]
        if folder is not None:
            path = os.path.join(folder, path)
        return path

    def get_single_example(self, idx: int) -> dict:
        sample, src = self.samples[idx]
        image_1 = self._resolve_path(sample, "path1", src)
        image_2 = self._resolve_path(sample, "path2", src)
        if not os.path.exists(image_1) or not os.path.exists(image_2):
            raise FileNotFoundError(f"Missing image: {image_1} or {image_2}")

        prompt = sample["prompt"]
        img1 = Image.open(image_1).convert("RGB").resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        img2 = Image.open(image_2).convert("RGB").resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        return {
            "image_1": image_1,
            "image_2": image_2,
            "pixel_value_image1": self.image_transforms(img1),
            "pixel_value_image2": self.image_transforms(img2),
            "text_1": prompt,
            "text_2": prompt,
        }

    def __getitem__(self, idx: int) -> dict:
        last_error = None
        tried = {idx}
        for _ in range(8):
            try:
                return self.get_single_example(idx)
            except Exception as exc:
                last_error = exc
                idx = random.randint(0, len(self.samples) - 1)
                if idx in tried and len(tried) >= min(8, len(self.samples)):
                    break
                tried.add(idx)
        raise RuntimeError(f"Failed to load sample after retries: {last_error}") from last_error

    @staticmethod
    def collate_fn(examples: List[dict]) -> dict:
        return {
            "pixel_values_1": torch.stack([ex["pixel_value_image1"] for ex in examples]),
            "pixel_values_2": torch.stack([ex["pixel_value_image2"] for ex in examples]),
            "text_1": [ex["text_1"] for ex in examples],
            "text_2": [ex["text_2"] for ex in examples],
            "image_1": [ex["image_1"] for ex in examples],
            "image_2": [ex["image_2"] for ex in examples],
        }
