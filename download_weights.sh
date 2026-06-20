#!/bin/bash
# Bake MuseTalk inference weights (~7.3GB) into the image. Run with cwd = the MuseTalk checkout
# (weights land in ./models, the paths handler.py / scripts.inference expect). Drops syncnet (training-only).
#
# Uses huggingface_hub's Python API (the Dockerfile pins hub==0.25.2). DO NOT `pip install -U
# huggingface_hub[cli]` here: hub 1.x makes the `huggingface-cli` command a no-op (silent empty download)
# AND breaks transformers 4.39.2. Both bit us in Phase 0; this path is the one that worked.
set -euo pipefail

python - <<'PY'
from huggingface_hub import hf_hub_download
jobs = [
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json",     "models"),
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth",          "models"),
    ("TMElyralab/MuseTalk", "musetalk/musetalk.json",        "models"),
    ("TMElyralab/MuseTalk", "musetalk/pytorch_model.bin",    "models"),
    ("stabilityai/sd-vae-ft-mse", "config.json",                 "models/sd-vae"),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", "models/sd-vae"),
    ("openai/whisper-tiny", "config.json",             "models/whisper"),
    ("openai/whisper-tiny", "pytorch_model.bin",       "models/whisper"),
    ("openai/whisper-tiny", "preprocessor_config.json","models/whisper"),
    ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", "models/dwpose"),
]
for repo, fn, ld in jobs:
    hf_hub_download(repo_id=repo, filename=fn, local_dir=ld)
    print("ok", repo, fn)
print("HF weights done")
PY

# Face parsing (BiSeNet) -- gdown from Google Drive + the torchvision resnet18 backbone.
mkdir -p models/face-parse-bisent
gdown "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812" -O models/face-parse-bisent/79999_iter.pth
curl -fsSL https://download.pytorch.org/models/resnet18-5c106cde.pth -o models/face-parse-bisent/resnet18-5c106cde.pth

echo "MuseTalk weights baked into ./models (~7.3GB)."
