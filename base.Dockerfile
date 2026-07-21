# ====================================================================================================
# vivijure-musetalk BASE image (2-image split) -- the stable, expensive half.
#
# Carries everything stable across a handler-only release: the cu128 base, the source-compiled openmmlab
# stack, the MuseTalk deps + checkout, and the ~7.3GB baked weights. The consumer image (Dockerfile) is
# just FROM this@digest + COPY handler.py + CMD, so a handler-only release re-pushes ONLY the handler
# layer; every base layer dedups on GHCR ("layer already exists"). That is the release win.
#
# SAME-PACKAGE by design: published as a TAG in the vivijure-musetalk package
# (ghcr.io/skyphusion-labs/vivijure-musetalk:base-<N>), NOT a separate -base package -- GHCR FROM-layer
# dedup on push is same-package only, so a separate package re-uploads every layer (S19 hard-won fact).
#
# Rebuilt ONLY on a deliberate dep/weight change via .github/workflows/base-build.yml (dispatch-only);
# a rebuild bumps base-<N> and the consumer repins its FROM digest. Contract: RELEASES.md.
# ====================================================================================================
# MuseTalk audio-driven lip-sync image -- GPU backend for Vivijure's `lipsync` module.
#
# MuseTalk (TMElyralab, MIT) rewrites a face video's mouth to match an audio track. Driven as a
# SUBPROCESS (python -m scripts.inference) -- the handler never links its internals.
#
# Base: RunPod's torch 2.8 / cu128 image -- VALIDATED end-to-end in Phase 0 on a live L40S (2026-06-20).
# Chosen over pytorch/pytorch:*-runtime because it ships the FULL CUDA 12.8 toolkit (nvcc), which mmcv's
# source build needs. It is py3.12, so the openmmlab stack needs the workarounds below -- every one
# proven on the pod, not guessed. See the README "Phase 0 recipe" section for why each line exists.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# TORCH_CUDA_ARCH_LIST = the GPU archs mmcv compiles CUDA ops for. RunPod serverless can hand us (or
# substitute) Ampere (A40/A10 = 8.6), Ada (L4/L40S = 8.9), Hopper (H100/H200 = 9.0), or Blackwell-Pro
# (RTX PRO 6000 = 12.0). A100 (8.0) and B200 (10.0) are left out -- we won't run a light module on those
# tiers; add them if a deploy lands there ("no kernel image is available for execution" = the running
# GPU's arch is missing from this list). MAX_JOBS bounds compile parallelism so the multi-arch build
# doesn't OOM the runner.
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0" \
    MAX_JOBS=4 \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk /app/MuseTalk

# py3.12 build fixes (Phase 0): setuptools 82 dropped pkg_resources and old setuptools lacks
# pkgutil.ImpImporter -- 75.6 has both. ninja/cython needed by the source builds below.
RUN pip install --no-cache-dir -U "setuptools==75.6.0" wheel ninja cython numpy pillow scipy matplotlib

# openmmlab stack. No prebuilt mmcv exists for cu128/torch2.8 (verified -- 404), so mmcv builds from
# source against torch 2.8 (nvcc present). mmpose/mmdet go in --no-deps to SKIP chumpy (py3.12-broken,
# 3D-body-only, unused) and pin the mmcv-2.1.0-compatible versions; their real deps are installed by
# hand. --no-build-isolation makes the old sdists use this env's setuptools, not pip's 82.
RUN pip install --no-cache-dir "mmengine>=0.10" && \
    MMCV_WITH_OPS=1 pip install --no-cache-dir --no-build-isolation "mmcv==2.1.0" && \
    pip install --no-cache-dir json_tricks munkres pycocotools terminaltables shapely && \
    pip install --no-cache-dir --no-build-isolation xtcocotools && \
    pip install --no-cache-dir --no-deps "mmpose==1.3.2" "mmdet==3.2.0" && \
    python -c "import mmcv,mmpose,mmdet; from mmcv.ops import nms; from mmpose.apis import init_model; print('mm stack OK', mmcv.__version__, mmpose.__version__, mmdet.__version__)"

# MuseTalk inference deps. huggingface_hub PINNED <1.0 -- hub 1.x makes huggingface-cli a no-op AND
# breaks transformers 4.39.2 (which requires hub <1.0). tensorflow/gradio dropped (unused at inference).
RUN pip install --no-cache-dir --ignore-installed cryptography \
      diffusers==0.30.2 accelerate==0.28.0 transformers==4.39.2 "huggingface_hub==0.25.2" \
      omegaconf librosa soundfile opencv-python einops "imageio[ffmpeg]" ffmpeg-python moviepy gdown \
      runpod boto3 pyyaml

# Bake all inference weights (~7.3GB) into the image -- no network volume. (Drops syncnet, training-only.)
COPY download_weights.sh /app/download_weights.sh
RUN cd /app/MuseTalk && bash /app/download_weights.sh

# Weights are baked, so go offline at runtime (no surprise HF fetches mid-job).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
