# DRM: Diffusion-based Reward Model

**CVPR 2026**

DRM turns a pretrained diffusion transformer (SD3.5 MMDiT) into a pairwise image reward model. Training uses Bradley-Terry loss on preference pairs, with a shared flow-matching timestep (`mix_time`) and independent noise on the win/lose latents.

This repository releases **training and inference code**. Official model weights are not included yet.

## Setup

```bash
git clone <this-repo>
cd DRM
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

You also need a local or Hugging Face copy of **Stable Diffusion 3.5 Medium** or **Large**, and pairwise training data in HPDv3 JSON format (`path1` is always the preferred image).

[HPDv3](https://huggingface.co/datasets/MizzenAI/HPDv3) is the dataset used in our experiments:

```bash
huggingface-cli download --repo-type dataset MizzenAI/HPDv3 --local-dir /path/to/HPDv3
```

Optional extra pairs (`pickapic.json`, `imagereward.json`) can be passed as additional `--train_json` files with matching `--image_folder` entries.

## Training

Official recipe:

- Freeze VAE and the three text encoders
- Train MMDiT + ConvGN reward head
- Resize images to a square (`--max_size`)
- Sample one timestep per pair (logit-normal by default; `--uniform` for uniform)
- Add independent flow-matching noise to win and lose latents
- Optimize \(-\log\sigma(s_{\text{win}} - s_{\text{lose}})\)

SD3.5 Medium (`2BConvGN`, skip last 3 blocks):

```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero2.yaml --num_processes 8 \
  drm/train.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --reward_head 2BConvGN \
  --max_size 512 \
  --train_json /path/to/HPDv3/train.json \
  --image_folder /path/to/HPDv3 \
  --confidence_threshold 0.95 \
  --train_batch_size 2 \
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --output_dir outputs/drm-sd35-medium
```

SD3.5 Large (`7BConvGN`, skip last 5 blocks):

```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero2.yaml --num_processes 8 \
  drm/train.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-large \
  --reward_head 7BConvGN \
  --max_size 512 \
  --train_json /path/to/HPDv3/train.json \
  --image_folder /path/to/HPDv3 \
  --uniform --shift \
  --gradient_checkpointing \
  --train_batch_size 1 \
  --output_dir outputs/drm-sd35-large
```

Example launch scripts: `scripts/train_sd35_medium.sh`, `scripts/train_sd35_large.sh`.

Each checkpoint directory is named `checkpoint-{step}` and contains `model.safetensors` (transformer + reward head) for inference.

To resume **optimizer state**, pass `--save_full_state` during training, then:

```bash
--resume_from_checkpoint latest
```

If the directory only has `model.safetensors`, resume still loads the weights and continues the step counter, but the optimizer is reinitialized.

T5 sequence length is **512** in both training and inference (same as the training script default).

## Inference

```python
import torch
from drm import DRMPipeline

pipe = DRMPipeline.from_sd3(
    "stabilityai/stable-diffusion-3.5-medium",
    reward_head="2BConvGN",
    image_size=512,
)
pipe.load_reward_model("outputs/drm-sd35-medium/checkpoint-2000/model.safetensors")
pipe.to(device="cuda", dtype=torch.float16)

score = pipe(image="assets/example.png", prompt="a photo of a cat")
print(score.item())
```

CLI:

```bash
python drm/infer.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --reward_head 2BConvGN \
  --reward_model_path outputs/drm-sd35-medium/checkpoint-2000/model.safetensors \
  --image /path/to/image.png \
  --prompt "a photo of a cat"
```

Typical evaluation scores the clean latent (`time=0`, no extra noise). That is the default.

## Pairwise evaluation

```bash
python eval/infer_pairwise.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --reward_head 2BConvGN \
  --reward_model_path /path/to/model.safetensors \
  --dataset_path /path/to/HPDv3/test.json \
  --image_folder /path/to/HPDv3 \
  --size 512 \
  --output_file results/eval.jsonl
```

Uses every visible GPU. Accuracy is `score(path1) > score(path2)`.

## Layout

```
drm/
  transformer.py   # SD3 MMDiT with intermediate-feature hooks
  model.py         # 2BConvGN / 7BConvGN
  dataset.py       # pairwise JSON
  pipeline.py      # encode + score
  train.py
  infer.py
eval/infer_pairwise.py
scripts/
accelerate_configs/
```

## License

Apache-2.0. The transformer implementation is derived from [Hugging Face Diffusers](https://github.com/huggingface/diffusers). See `NOTICE`.

SD3.5 weights are subject to Stability AI's license; you must obtain them separately.
