# MuseTalk audio-driven lip-sync image -- GPU backend for Vivijure's `lipsync` module.
#
# MuseTalk (TMElyralab, MIT) rewrites a face video's mouth to match an audio track. Run as a SUBPROCESS
# (python -m scripts.inference) -- the handler never links its internals.
#
# Base MUST be torch 2.7+ / CUDA 12.8 (cu128), same as vivijure-upscale: RunPod's fleet includes
# Blackwell cards (sm_120) and substitutes them even for smaller requests; older cu121 torch ships no
# kernels for them -> "no kernel image is available" at CUDA init -> the worker won't boot. cu128 carries
# the sm_100 (B200) + sm_120 (Blackwell) kernels. NOTE: upstream MuseTalk targets torch 2.0.1, so the
# diffusers/transformers pins (requirements.txt) and the mmcv/mmpose build below are the two things the
# Phase-0 pod validates on this newer torch before we trust the image.

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# MuseTalk source (MIT) -- pinned at build; bumped deliberately, not floating.
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk /app/MuseTalk

# Python deps (MuseTalk's, minus gradio, plus the module runtime).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# mmpose/mmcv drive MuseTalk's dwpose face-landmark step. They are NOT in MuseTalk's requirements and
# are torch/cuda-version-sensitive -- THIS is the known integration risk on the cu128 bump (mmcv prebuilt
# wheels lag torch, so mim may build mmcv from source against torch 2.8). Validated in Phase 0; if the
# source build is slow/fragile, the fallback is pinning a known-good (torch, mmcv) pair.
RUN pip install --no-cache-dir -U openmim && \
    mim install "mmengine>=0.10" "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"

# Bake all inference weights (~5GB) into the image -- no network volume, same baked-model approach as
# the rest of the stack. (Drops syncnet, which is training-only.)
COPY download_weights.sh /app/download_weights.sh
RUN cd /app/MuseTalk && bash /app/download_weights.sh

# Weights are baked, so go offline at runtime (no surprise HF fetches mid-job).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "handler.py"]
