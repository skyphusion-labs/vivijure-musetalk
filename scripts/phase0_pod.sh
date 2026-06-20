#!/bin/bash
# Phase 0 -- prove MuseTalk runs on the cu128 stack BEFORE building the endpoint.
#
# Run this on a throwaway RunPod POD started from `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
# (cheap, minutes). It mirrors the Dockerfile steps interactively so the known risks surface fast:
#   1. does `mim install mmcv` succeed against torch 2.8 (or must it build from source / get pinned)?
#   2. do diffusers/transformers behave on torch 2.8?
#   3. does MuseTalk actually produce a good lip-synced clip from its own sample?
# Then re-run the final step on an ANIMATED clip to answer: does MuseTalk generalize past photoreal?
set -euo pipefail

apt-get update && apt-get install -y --no-install-recommends git ffmpeg ca-certificates curl

[ -d MuseTalk ] || git clone --depth 1 https://github.com/TMElyralab/MuseTalk
cd MuseTalk

# MuseTalk deps (upstream minus gradio) + the dwpose stack via openmim (the cu128 risk).
pip install --no-cache-dir \
  diffusers==0.30.2 accelerate==0.28.0 numpy==1.23.5 tensorflow==2.12.0 tensorboard==2.12.0 \
  opencv-python==4.9.0.80 soundfile==0.12.1 transformers==4.39.2 huggingface_hub==0.30.2 \
  librosa==0.11.0 einops==0.8.1 omegaconf ffmpeg-python moviepy "imageio[ffmpeg]" gdown
pip install --no-cache-dir -U openmim
mim install "mmengine>=0.10" "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"

# Weights (~5GB). Uses this repo's adapted script if present, else upstream's.
if [ -f /workspace/download_weights.sh ]; then bash /workspace/download_weights.sh
else bash download_weights.sh; fi

# The actual proof: run V1.5 inference on MuseTalk's own sample (real face + speech).
python -m scripts.inference \
  --inference_config configs/inference/test.yaml \
  --result_dir results/test \
  --unet_model_path models/musetalkV15/unet.pth \
  --unet_config models/musetalkV15/musetalk.json \
  --version v15

echo "=== Phase 0 output ==="; ls -laR results/test
echo "Pull a result mp4 and eyeball the sync. Then drop an ANIMATED clip into a config + rerun to test generalization."
