#!/bin/bash
# Bake MuseTalk inference weights into the image. Run with cwd = the MuseTalk checkout (weights land in
# ./models, matching the relative paths the handler passes to scripts.inference).
#
# Adapted from upstream download_weights.sh: drops syncnet (training-only, not used at inference) and
# uses the DEFAULT HuggingFace endpoint (upstream forces the hf-mirror.com CN mirror -- slow/unreliable
# from Hetzner/RunPod). Total ~5GB baked (no network volume, same approach as the rest of the stack).
set -euo pipefail
Ck="models"
mkdir -p "$Ck/musetalk" "$Ck/musetalkV15" "$Ck/dwpose" "$Ck/face-parse-bisent" "$Ck/sd-vae" "$Ck/whisper"
pip install --no-cache-dir -U "huggingface_hub[cli]" gdown

# MuseTalk V1.0 + V1.5 UNet (V1.5 = default/best)
huggingface-cli download TMElyralab/MuseTalk --local-dir "$Ck" \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin" \
            "musetalkV15/musetalk.json" "musetalkV15/unet.pth"
# SD VAE (ft-mse) -- the latent decoder
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir "$Ck/sd-vae" \
  --include "config.json" "diffusion_pytorch_model.bin"
# Whisper-tiny -- audio encoder
huggingface-cli download openai/whisper-tiny --local-dir "$Ck/whisper" \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"
# DWPose -- face landmarks (needs mmpose/mmcv at runtime)
huggingface-cli download yzd-v/DWPose --local-dir "$Ck/dwpose" \
  --include "dw-ll_ucoco_384.pth"
# Face parsing (BiSeNet) -- mouth-region mask for blending
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O "$Ck/face-parse-bisent/79999_iter.pth"
curl -fsSL https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o "$Ck/face-parse-bisent/resnet18-5c106cde.pth"

echo "MuseTalk weights baked into $Ck."
