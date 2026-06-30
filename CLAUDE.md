# CLAUDE.md

Guidance for Claude Code (and the crew) working in this repo.

## What this is

**The GPU backend for Vivijure's `lipsync` finish module.** A single RunPod serverless image that
lip-syncs a face video to a dialogue track with **MuseTalk** (TMElyralab, MIT). It is the finish-class
brick that gives the films talking characters whose mouths match the speech: it runs **after i2v**, on
dialogue / close shots with a clear face in frame. Pairs with `vivijure-upscale` (MuseTalk works a
256x256 face region, so a synced shot should pass back through Real-ESRGAN to return to delivery res).

This repo is the image + the RunPod handler; the studio-side `lipsync` module worker (a thin CF Worker
behind the typed finish hook in `vivijure`) is what calls this endpoint. Image:
`ghcr.io/skyphusion-labs/vivijure-musetalk` (current release tag **v0.1.0**, the immutable tag the
endpoint pins to).

## The Vivijure constellation (the same map is in each repo)

```
   friends + Slate (Discord)
            |
            v
        slate  -->  vivijure (studio control plane / JSON API)
                        |
                        v
                  vivijure-backend (GPU render: keyframes -> i2v -> assemble)
                        |
            +-----------+-------------+-------------------+
            |           |             |                   |
   vivijure-musetalk  vivijure-   vivijure-audio-   vivijure-local-backend
   (lipsync module)   upscale     upscale           (self-host render path)
       ^-- THIS REPO
```

## Handler contract (the job, `handler.py`)

One typed in / one typed out, three dispatch modes (`handler(job)` branches on the input keys). The one
difference vs a one-video module: **two inputs**, a face clip AND an audio track.

- **R2 finish-chain mode** (the endpoint reads/writes the shared bucket itself, no creds on the wire):
  `{ clip_key, audio_key, output_key?, bbox_shift?, version? }`. Returns the new key as `clip_key` so
  the finish chain carries the synced clip downstream.
- **Presigned mode** (credentialless: the core presigns R2): `{ video_url, audio_url, output_url, output_key }`.
- **Selftest:** `{ "selftest": true }` runs MuseTalk end to end on its OWN baked sample (a real face +
  speech; a synthetic testsrc has no face to detect), proving CUDA + the full UNet/VAE/whisper +
  dwpose/face-parse stack. Doubles as the endpoint health check.

Success returns `{ ok, clip_key|output_key, bytes, version, applied:["lipsync:<ver>"] }`. **A non-ok
result is a soft-degrade** the module honors by passing the original clip through untouched, never a
drop: a shot with no detectable face must come back unchanged, not fail the render. The `applied` tag
is set only on a real sync.

**Two contract invariants to preserve:**
- `version` is `v15` (default, best) or `v1`; MuseTalk is driven as a SUBPROCESS (`python -m
  scripts.inference` against a temp config yaml), so this handler never imports MuseTalk's internals
  (clean process boundary; its dep tree stays isolated).
- **A synced shot is the same duration as the face clip that went in** (`_pad_audio_to_video`).
  MuseTalk's output length follows the AUDIO, so a short dialogue line over a long shot would truncate
  the clip; the handler pads the audio with trailing silence to the face-clip duration first. The pad
  is best-effort (ffprobe/ffmpeg failure falls back to the original audio, never worse than un-padded).

## Commands

This is a Python / RunPod image, NOT an npm package. There is no local test suite; verification is the
build-time fail-fasts plus the GPU-gated selftest.

```bash
# Build the image locally (the CUDA base + mmcv source build is large + slow; CI does this on push).
docker build -t vivijure-musetalk:dev .

# Lint the handler without a GPU.
python -m py_compile handler.py

# GPU verify (on a pinned RunPod endpoint, or a live GPU pod): send {"selftest": true} and assert ok:true.
```

**Release / deploy mechanics.** `.github/workflows/build-image.yml` builds + pushes to GHCR on a push
to `main` (touching the build inputs) as `:latest` + `:<sha>`; a pushed semver tag (`v0.1.0`) ALSO
publishes the bare `:0.1.0` (the immutable tag the endpoint pins to, never `:sha`, never `:latest`).
PUBLIC repo, so CI runs on GitHub-hosted `ubuntu-latest` (the "Free disk space" step reclaims room for
the tens-of-GB CUDA build). The RunPod endpoint's image tag, **GPU type, and R2 env are dashboard /
endpoint-config knobs** (RunPod's API does not honor them); **container-registry-auth IS now
MCP/API-manageable** (RunPod MCP `create-container-registry-auth` + attach via `containerRegistryAuthId`
on create/update-template, no dashboard step). `scripts/phase0_pod.sh` is the live-pod bring-up used to
validate the recipe.

## Verifying changes

The recipe is fragile by nature (py3.12 + cu128/torch2.8 + an openmmlab source build). Every Dockerfile
line is proven on a live pod, not guessed; `requirements.txt` is a human-readable manifest of the proven
pin set, NOT a flat `pip install -r` (order + `--no-deps` / `--no-build-isolation` / `MMCV_WITH_OPS` /
the version pins matter and a flat install will not reproduce a working env). After any dependency or
Dockerfile change: build clean, then run `{"selftest": true}` on a real GPU and confirm `ok:true` with a
non-zero `output_bytes` before cutting a release tag. `TORCH_CUDA_ARCH_LIST` here is REAL (mmcv compiles
CUDA ops): `8.6;8.9;9.0;12.0`. A missing arch is "no kernel image is available for execution" at runtime.

## Architecture

- **Subprocess boundary.** The handler shells out to MuseTalk; it owns transport (R2 / presigned),
  the audio-pad fix, temp-dir hygiene, and the result envelope, nothing of MuseTalk's model code.
- **Baked weights, no network volume.** ~7.3GB of inference weights (MuseTalk V1.5 + V1.0 UNet,
  sd-vae-ft-mse, whisper-tiny, DWPose, face-parse BiSeNet) are baked into the image via
  `download_weights.sh`; runtime is `HF_HUB_OFFLINE=1` so no surprise HF fetch mid-job. syncnet is
  dropped (training-only). This is the GPU-rationing thesis: scale-to-zero, no cold-pull.
- **Perf follow-up (deferred).** The subprocess reloads ~5GB per job, throwing away warm-worker state.
  The optimization is the upscale module's in-process `_MODELS` warm-cache pattern; deferred until the
  model + dep set are proven.

## Conventions

- **No em-dashes (U+2014) or en-dashes (U+2013) anywhere.** Use commas, semicolons, parentheses, or `--`.
- Handle / username is `skyphusion` across all services.
- **Honest soft-degrade is the contract** (the #245 / #249 discipline): on a polish miss return the
  passthrough, NO fake `applied` tag, never fail the chain; only malformed I/O fails loud.
- Minimal deps; the pin set is sacred (see Verifying changes). Justify and re-validate any change to it.
- MuseTalk (MIT) + its weights are redistributed under their upstream licenses; the full inventory is
  `THIRD_PARTY_NOTICES.md`. Keep it current when a bundled upstream changes.

## Crew + identity

- The FIRST command in any op is the member's own login shell: `sudo -u <member> bash -lc '<ops>'`
  (loads their `$HOME`, their `~/dev/vivijure-musetalk` clone, their gh / RunPod / R2 creds). Commits
  and PRs land under the member's `skyphusion-<member>` identity, never Conrad's.
- Operating memory for the vivijure family lives in the per-project memory under
  `~/.claude/projects/-home-conrad-dev-vivijure/memory/` (`seg-vivijure-modules` is the relevant
  segment); load it before acting.
- **HARD AUP line:** the CSAM bright line is absolute (see the vivijure project memory). Non-negotiable.

## Commits & versioning

Conventional Commits (`feat(scope):`, `fix(scope):`, `docs:`); body explains the why. SemVer-style
`0.MINOR.PATCH` while pre-1.0 (PATCH for fixes / backend tweaks, MINOR for features). A release is a
pushed `vMAJOR.MINOR.PATCH` git tag (CI publishes the matching immutable image tag the endpoint pins to).
