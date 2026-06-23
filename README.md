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

## Lip-sync keeps the full clip length (#6)

MuseTalk's output length follows the **audio** track, not the face clip. When a shot's dialogue is
shorter than the i2v clip (the common case: a 1.4s line over a 5s shot), upstream MuseTalk emits
only the talking segment and **truncates the shot to the spoken-line length**. In a scatter talking
film that surfaced as a clip-drop -- a 5s shot came back 1.4s and the assembled film ran short.

The handler fixes this **before** inference (`_pad_audio_to_video` in `handler.py`): it ffprobes the
face clip and the audio, and if the audio is shorter it pads the audio with **trailing silence
(`apad -t <clip_seconds>`)** to the face-clip duration. MuseTalk then renders the full clip -- the
line is spoken at the head, the mouth rests (closed) for the remainder -- so the synced shot keeps
its **full original length**. The pad is best-effort: if ffprobe/ffmpeg fails it falls back to the
original audio (never worse than the un-padded behavior).

Net contract: **a lip-synced shot is the same duration as the face clip that went in.** A downstream
concat/gather can rely on per-shot durations being preserved end to end.

## Weights (baked, ~5GB, no volume)
MuseTalk V1.5 + V1.0 UNet, sd-vae-ft-mse, whisper-tiny, DWPose, face-parse BiSeNet. See
`download_weights.sh` (adapted from upstream: syncnet dropped, default HF endpoint).

## License boundary
MuseTalk is MIT -- no process-isolation *requirement*, but we subprocess it anyway for a clean dep
boundary. This image redistributes MuseTalk + its model weights under their respective upstream licenses.

## License

**AGPL-3.0-only.** A labor of love, given freely: use it, learn from it, self-host it, build your own creative visions on it. Run it as a network service and the AGPL has you share your changes back, so it stays a commons. It is not for sale, and not to be resold as a SaaS.

It redistributes **MuseTalk** (MIT, TMElyralab) and its model weights under their respective upstream licenses (see "License boundary" above). A full third-party license inventory -- every bundled upstream, its source, and its license text -- is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
