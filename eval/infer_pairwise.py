#!/usr/bin/env python
"""Pairwise preference accuracy on HPDv3-style JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from drm.dataset import PairwiseRewardDataset
from drm.pipeline import DRMPipeline


def worker_process(device_id, task_queue, result_queue, args):
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    pipeline = DRMPipeline.from_sd3(
        args.pretrained_model_name_or_path,
        reward_head=args.reward_head,
        image_size=args.size,
        max_sequence_length=args.max_sequence_length,
    )
    pipeline.load_reward_model(args.reward_model_path, device=device, dtype=dtype)
    pipeline.to(device=device, dtype=dtype)

    test_dataset = PairwiseRewardDataset(
        json_list=[args.dataset_path],
        image_folder=[args.image_folder] if args.image_folder else None,
        confidence_threshold=None,
        image_size=args.size,
    )

    while True:
        task_index = task_queue.get()
        if task_index is None:
            break
        try:
            batch = test_dataset[task_index]
            pixel_values_1 = batch["pixel_value_image1"].unsqueeze(0).to(device, dtype=dtype)
            pixel_values_2 = batch["pixel_value_image2"].unsqueeze(0).to(device, dtype=dtype)
            prompts = [batch["text_1"]]
            with torch.no_grad():
                score_win = pipeline(
                    image=pixel_values_1, prompt=prompts, time=args.time, add_noise=args.add_noise
                ).item()
                score_lose = pipeline(
                    image=pixel_values_2, prompt=prompts, time=args.time, add_noise=args.add_noise
                ).item()
            result_queue.put(
                {
                    "path_1": batch["image_1"],
                    "path_2": batch["image_2"],
                    "win": score_win,
                    "lose": score_lose,
                }
            )
        except Exception as exc:
            print(f"Error on GPU {device_id} index {task_index}: {exc}")
            result_queue.put({"error": str(exc), "index": task_index})


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DRM pairwise accuracy.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="results/eval.jsonl")
    parser.add_argument("--reward_head", type=str, default="2BConvGN", choices=["2BConvGN", "7BConvGN"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--max_sequence_length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mp.set_start_method("spawn", force=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found.")

    probe_dataset = PairwiseRewardDataset(
        json_list=[args.dataset_path],
        image_folder=[args.image_folder] if args.image_folder else None,
        confidence_threshold=None,
        image_size=args.size,
    )
    num_tasks = len(probe_dataset)
    print(f"Found {num_gpus} GPUs, {num_tasks} pairs.")

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for i in range(num_tasks):
        task_queue.put(i)
    for _ in range(num_gpus):
        task_queue.put(None)

    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=worker_process, args=(i, task_queue, result_queue, args))
        p.start()
        processes.append(p)

    all_results = []
    failed = []
    for _ in tqdm(range(num_tasks), desc="Evaluating"):
        result = result_queue.get()
        if result is None or "error" in result:
            failed.append(result)
        else:
            all_results.append(result)

    for p in processes:
        p.join()

    if failed:
        print(f"Warning: {len(failed)} / {num_tasks} pairs failed.")
        for item in failed[:10]:
            print(f"  failed: {item}")

    correct = sum(1 for r in all_results if r["win"] > r["lose"])
    total = len(all_results)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    print("=" * 50)
    print(f"Processed: {total}")
    print(f"Failed:    {len(failed)}")
    print(f"Correct:   {correct}")
    if total:
        print(f"Accuracy:  {correct / total * 100:.2f}% (over successful pairs)")
    else:
        print("Accuracy:  n/a (no successful pairs)")
    print(f"Saved:     {args.output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
