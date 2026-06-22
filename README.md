# vivijure-musetalk

A RunPod serverless image that lip-syncs a face video to an audio track with
[MuseTalk](https://github.com/TMElyralab/MuseTalk) (TMElyralab, **MIT**). The GPU backend for
Vivijure's `lipsync` module: the finish-class brick that gives the films **talking characters** whose
mouths match the dialogue.

It runs **after i2v**, on dialogue / close shots that have a clear face in frame. It is the natural
partner to the upscaler: MuseTalk works a 256x256 face region, so a lip-synced shot should pass through
the Real-ESRGAN upscaler ([vivijure-upscale](https://github.com/skyphusion-labs/vivijure-upscale)) to
return to delivery resolution.

## How it fits the stack
Same shape as `vivijure-upscale` -- a single-purpose, baked-weights, scale-to-zero endpoint, **separate**
from the heavy `vivijure-backend`. The handler drives MuseTalk as a **subprocess**
(`python -m scripts.inference`), exactly like the upscale module subprocesses ffmpeg/video2x, so this
code never imports MuseTalk's internals (clean process boundary; MuseTalk's dep tree stays isolated).

## Handler contract (job input)
Three modes, identical in spirit to the rest of the module stack. The one difference vs a one-video
module: **two inputs**, a face clip AND an audio track.

**R2 finish-chain mode** (the endpoint reads/writes the shared bucket):
```json
{
  "clip_key":   "renders/<project>/clips/<shot>.mp4",
  "audio_key":  "renders/<project>/audio/<shot>.wav",
  "output_key": "renders/<project>/clips/<shot>_ls.mp4",
  "bbox_shift": 0,
  "version":    "v15"
}
```
**Presigned mode** (credentialless -- the core presigns R2): `{ video_url, audio_url, output_url, output_key }`.

**Selftest:** `{ "selftest": true }` runs MuseTalk end to end on its **own baked sample** (a real face +
speech -- a synthetic testsrc has no face to detect), confirming CUDA + the full UNet/VAE/whisper +
dwpose/face-parse stack. Doubles as the endpoint health check.

Returns `{ ok, clip_key|output_key, bytes, version, applied:["lipsync:v15"] }`. **A non-ok result is a
soft-degrade** -- the module passes the original clip through untouched, never a drop. A shot with no
detectable face must come back unchanged, not fail the render (so the upstream face gate can be
best-effort: a misroute degrades, it doesn't break).

## Weights (baked, ~5GB, no volume)
MuseTalk V1.5 + V1.0 UNet, sd-vae-ft-mse, whisper-tiny, DWPose, face-parse BiSeNet. See
`download_weights.sh` (adapted from upstream: syncnet dropped, default HF endpoint).

## Phase 0 -- DONE (validated on a live L40S, 2026-06-20)
MuseTalk runs end-to-end on the cu128/torch2.8 stack: v1.5 inference produced a clean lip-synced clip
(704x1216, 25fps, h264 + AAC, 8s) from the stock sample. The full proven recipe is baked into the
Dockerfile. Notably, MuseTalk's own stock sample is a **stylized character** and it lip-syncs cleanly --
an early yes to "does it handle animation."

The non-obvious things Phase 0 settled (each is a line in the Dockerfile):
- **No prebuilt mmcv for cu128/torch2.8** (404 on the OpenMMLab index; cu121/torch2.1 is the newest). So
  mmcv is **built from source** against torch 2.8 -- needs nvcc, which is why the base ships the full CUDA
  toolkit. mmcv 2.1.0 (matches mmpose 1.3.2 / mmdet 3.2.0).
- **py3.12 packaging traps:** setuptools 82 dropped `pkg_resources` and old setuptools lacks
  `pkgutil.ImpImporter` -> pin setuptools 75.6; old sdists (mmcv, xtcocotools) need `--no-build-isolation`.
- **chumpy** (a transitive mmpose dep, 3D-body-only, unused) is py3.12-broken -> install mmpose/mmdet
  `--no-deps` and hand-install their real deps.
- **huggingface_hub must stay <1.0** -- hub 1.x makes `huggingface-cli` a no-op (silent empty download)
  and breaks transformers 4.39.2.
- **torch 2.6+ `weights_only=True` default** breaks loading the numpy-pickled checkpoints -> set
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

Remaining before ship: build + push the image (Phase 2), stand up the endpoint (Phase 3), wire into the
control plane (Phase 4), one real shot end-to-end (Phase 5). One caveat: `TORCH_CUDA_ARCH_LIST` is broad
(sm_80..sm_120) but Phase 0's test build compiled arch 8.9 only, so the broad mmcv build is the single
step not yet exercised.

## Roadmap / follow-ups
- **Warm model cache.** Subprocess reloads ~5GB of models per job, wasting warm-worker state. The
  optimization is an in-process warm cache (the upscale module's `_MODELS` pattern). Deferred until
  Phase 0 proves the model + dep set.
- **Upstream face gate.** The control plane should route only dialogue/close shots with a clear face;
  the module's soft-degrade is the safety net, not the primary gate.

## License boundary
MuseTalk is MIT -- no process-isolation *requirement*, but we subprocess it anyway for a clean dep
boundary. This image redistributes MuseTalk + its model weights under their respective upstream licenses.

## License

This repo is [AGPL-3.0](LICENSE) -- free as in yours: self-host it, learn from it, build on it.

It redistributes **MuseTalk** (MIT, TMElyralab) and its model weights under their respective upstream
licenses (see "License boundary" above). A full third-party license inventory -- every bundled
upstream, its source, and its license text -- is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
