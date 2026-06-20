"""RunPod serverless handler -- MuseTalk audio-driven lip-sync for Vivijure's `lipsync` module.

MuseTalk (TMElyralab, MIT) rewrites the mouth region of a face VIDEO to match an AUDIO track. This is
the finish-class brick that gives Vivijure talking characters: it runs AFTER i2v, on dialogue / close
shots that have a clear face in frame.

Same transport contract + {"selftest": true} harness as the rest of the module stack (mirrors
vivijure-upscale). The one real difference: TWO inputs (a face clip AND an audio track) instead of one.

We drive MuseTalk as a SUBPROCESS (`python -m scripts.inference` against a temp config yaml), the same
way the upscale module subprocesses ffmpeg/video2x -- so this handler never imports MuseTalk's internals
(clean process boundary; MuseTalk's deps stay isolated from ours).

Job input (R2 finish-chain mode -- the endpoint reads/writes the shared bucket itself):
  {
    "clip_key":   "renders/<project>/clips/<shot>.mp4",      # required -- the face video
    "audio_key":  "renders/<project>/audio/<shot>.wav",      # required -- the dialogue to sync to
    "output_key": "renders/<project>/clips/<shot>_ls.mp4",   # optional -- defaults to <clip>_ls.mp4
    "bbox_shift": 0,          # optional MuseTalk mouth-region tuning (+ opens / - closes the crop)
    "version":    "v15"       # v15 (default, best) | v1
  }

Job input (presigned mode -- credentialless handler, the core presigns R2):
  { "video_url": "<GET clip>", "audio_url": "<GET audio>", "output_url": "<PUT result>", "output_key": "..." }

Returns: { ok, clip_key|output_key, bytes, version, applied:["lipsync:<ver>"] } on success;
{ ok: false, error } otherwise. A non-ok result is a SOFT-DEGRADE -- the lipsync module passes the
original clip through untouched, never a drop (a shot with no detectable face must come back unchanged,
not fail the render).

PERF FOLLOW-UP (post Phase 0): subprocess reloads ~5GB of models per job, throwing away warm-worker
state. The optimization is to import MuseTalk's model-load once into a warm cache (the upscale module's
`_MODELS` pattern) and call inference in-process. Deferred until Phase 0 proves the model + the dep set.
"""

import glob
import os
import shutil
import subprocess
import tempfile

import boto3
import requests
import runpod
import torch
import yaml

MUSETALK_DIR = os.environ.get("MUSETALK_DIR", "/app/MuseTalk")
DOWNLOAD_TIMEOUT = 900
UPLOAD_TIMEOUT = 900

# UNet weights, relative to MUSETALK_DIR (scripts.inference runs with cwd=MUSETALK_DIR).
UNET = {
    "v15": ("models/musetalkV15/unet.pth", "models/musetalkV15/musetalk.json"),
    "v1": ("models/musetalk/pytorch_model.bin", "models/musetalk/musetalk.json"),
}

R2_ENDPOINT = os.environ.get("R2_ENDPOINT_URL", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "vivijure")


def _r2():
    return boto3.client(
        "s3", endpoint_url=R2_ENDPOINT, region_name="auto",
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
    )


def _get(url, dst):
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def _run_musetalk(face_path, audio_path, out_path, bbox_shift=0, version="v15"):
    """Write a temp MuseTalk config (one task: face video + audio), run scripts.inference, and move the
    produced clip to out_path. Output naming varies by version, so we glob the newest mp4 in result_dir."""
    version = version if version in UNET else "v15"
    unet_path, unet_cfg = UNET[version]
    cfg_dir = tempfile.mkdtemp(prefix="ms-cfg-")
    result_dir = tempfile.mkdtemp(prefix="ms-out-")
    cfg_path = os.path.join(cfg_dir, "task.yaml")
    task = {"task_0": {"video_path": face_path, "audio_path": audio_path}}
    if bbox_shift:
        task["task_0"]["bbox_shift"] = int(bbox_shift)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(task, f)
    cmd = ["python3", "-m", "scripts.inference",
           "--inference_config", cfg_path,
           "--result_dir", result_dir,
           "--unet_model_path", unet_path,
           "--unet_config", unet_cfg,
           "--version", version]
    try:
        p = subprocess.run(cmd, cwd=MUSETALK_DIR, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"musetalk inference rc={p.returncode}: {(p.stderr or p.stdout or '')[-800:]}")
        mp4s = sorted(glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True),
                      key=os.path.getmtime)
        if not mp4s:
            raise RuntimeError("musetalk produced no output mp4")
        shutil.move(mp4s[-1], out_path)
        return out_path
    finally:
        shutil.rmtree(cfg_dir, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)


def _selftest(inp):
    """Self-contained GPU verification -- NO R2 needed. Runs MuseTalk end to end on its OWN baked sample
    (a real face + speech; a synthetic testsrc has no face to detect). Confirms CUDA + the full model +
    dwpose/face-parse + VAE + whisper stack. Trigger with {"selftest": true}; doubles as a health check."""
    out = {"ok": False, "selftest": True, "torch_version": torch.__version__,
           "cuda_available": torch.cuda.is_available()}
    version = str(inp.get("version", "v15"))
    work = tempfile.mkdtemp(prefix="selftest-")
    dst = os.path.join(work, "out.mp4")
    try:
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
        face = os.path.join(MUSETALK_DIR, "data/video/yongen.mp4")
        audio = os.path.join(MUSETALK_DIR, "data/audio/yongen.wav")
        out["sample_present"] = os.path.exists(face) and os.path.exists(audio)
        if not out["sample_present"]:
            out["error"] = "baked sample missing (data/video/yongen.mp4 + data/audio/yongen.wav)"
            return out
        _run_musetalk(face, audio, dst, version=version)
        if not os.path.exists(dst) or not os.path.getsize(dst):
            out["error"] = "no output produced"
            return out
        out["output_bytes"] = os.path.getsize(dst)
        out["version"] = version
        out["ok"] = True
        return out
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        out["error"] = str(e)[:800]
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _lipsync_r2(inp):
    """R2 mode: download clip_key + audio_key, lip-sync, upload output_key in the shared bucket; return
    the new key as `clip_key` so the finish chain carries the lip-synced clip downstream."""
    clip_key = inp.get("clip_key")
    audio_key = inp.get("audio_key")
    if not audio_key:
        return {"ok": False, "error": "lipsync needs both clip_key and audio_key"}
    name = clip_key.rsplit("/", 1)[-1]
    output_key = inp.get("output_key") or (
        f"{clip_key.rsplit('.', 1)[0]}_ls.{clip_key.rsplit('.', 1)[1]}" if "." in name else f"{clip_key}_ls")
    bbox_shift = int(inp.get("bbox_shift", 0) or 0)
    version = str(inp.get("version", "v15"))
    if not (R2_ENDPOINT and os.environ.get("R2_ACCESS_KEY_ID")):
        return {"ok": False, "error": "R2 mode needs R2_ENDPOINT_URL + R2_ACCESS_KEY_ID/SECRET in the endpoint env"}
    work = tempfile.mkdtemp(prefix="ls-")
    face = os.path.join(work, "face.mp4")
    audio = os.path.join(work, "audio.wav")
    dst = os.path.join(work, "out.mp4")
    try:
        s3 = _r2()
        s3.download_file(R2_BUCKET, clip_key, face)
        s3.download_file(R2_BUCKET, audio_key, audio)
        _run_musetalk(face, audio, dst, bbox_shift=bbox_shift, version=version)
        if not os.path.getsize(dst):
            return {"ok": False, "error": "lipsync produced no output"}
        s3.upload_file(dst, R2_BUCKET, output_key, ExtraArgs={"ContentType": "video/mp4"})
        return {"ok": True, "clip_key": output_key, "bytes": os.path.getsize(dst),
                "version": version, "applied": [f"lipsync:{version}"]}
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        return {"ok": False, "error": str(e)[:500]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _lipsync_presigned(inp):
    """Presigned mode: GET video_url + audio_url, lip-sync, PUT output_url. Credentialless -- no R2 creds
    cross the wire."""
    video_url = inp.get("video_url")
    audio_url = inp.get("audio_url")
    output_url = inp.get("output_url")
    output_key = inp.get("output_key", "")
    if not (video_url and audio_url and output_url):
        return {"ok": False, "error": "input needs presigned video_url + audio_url + output_url"}
    bbox_shift = int(inp.get("bbox_shift", 0) or 0)
    version = str(inp.get("version", "v15"))
    work = tempfile.mkdtemp(prefix="ls-")
    face = os.path.join(work, "face.mp4")
    audio = os.path.join(work, "audio.wav")
    dst = os.path.join(work, "out.mp4")
    try:
        _get(video_url, face)
        _get(audio_url, audio)
        _run_musetalk(face, audio, dst, bbox_shift=bbox_shift, version=version)
        size = os.path.getsize(dst)
        if not size:
            return {"ok": False, "error": "lipsync produced no output"}
        with open(dst, "rb") as f:
            put = requests.put(output_url, data=f, timeout=UPLOAD_TIMEOUT,
                               headers={"content-type": "video/mp4", "content-length": str(size)})
        put.raise_for_status()
        return {"ok": True, "output_key": output_key, "bytes": size,
                "version": version, "applied": [f"lipsync:{version}"]}
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        return {"ok": False, "error": str(e)[:500]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def handler(job):
    inp = (job or {}).get("input") or {}
    if inp.get("selftest"):
        return _selftest(inp)
    if inp.get("clip_key"):
        return _lipsync_r2(inp)
    return _lipsync_presigned(inp)


runpod.serverless.start({"handler": handler})
