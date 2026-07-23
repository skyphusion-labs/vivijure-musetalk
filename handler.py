"""RunPod serverless handler -- MuseTalk audio-driven lip-sync for Vivijure's `lipsync` module.

MuseTalk (TMElyralab, MIT) rewrites the mouth region of a face VIDEO to match an AUDIO track. This is
the finish-class brick that gives Vivijure talking characters: it runs AFTER i2v, on dialogue / close
shots that have a clear face in frame.

Same transport contract + {"selftest": true} harness as the rest of the module stack (mirrors
vivijure-upscale). The one real difference: TWO inputs (a face clip AND an audio track) instead of one.

We drive MuseTalk IN-PROCESS: the ~5GB model set (vae/unet/pe + whisper + face-parse) is loaded once
per warm RunPod worker and cached (the upscale module's `_MODELS` pattern), then every job reuses it.
This replaced the old `python -m scripts.inference` subprocess, which reloaded all ~5GB from disk onto
the GPU on every single job and discarded the warm state (GPU-billed seconds burned per job). MuseTalk's
inference helpers (`load_all_model`, `AudioProcessor`, `FaceParsing`, `datagen`, `get_image`, ...) are the
same public surface its own CLI (`scripts.inference`) drives; we import them lazily inside `_pipeline`, so
a missing/broken MuseTalk checkout surfaces as a job error (honest soft-degrade), not a worker-boot crash.

Job input (R2 finish-chain mode -- the endpoint reads/writes the shared bucket itself):
  {
    "project":    "<project>",                               # required -- scopes every renders/ key
    "clip_key":   "renders/<project>/clips/<shot>.mp4",      # required -- the face video
    "audio_key":  "renders/<project>/audio/<shot>.wav",      # required -- the dialogue to sync to
    "output_key": "renders/<project>/clips/<shot>_ls.mp4",   # optional -- defaults to <clip>_ls.mp4
    "bbox_shift": 0,          # optional MuseTalk mouth-region tuning (+ opens / - closes the crop; v1 only)
    "version":    "v15"       # v15 (default, best) | v1
  }

Job input (presigned mode -- credentialless handler, the core presigns R2):
  { "video_url": "<GET clip>", "audio_url": "<GET audio>", "output_url": "<PUT result>", "output_key": "..." }

Returns: { ok, clip_key|output_key, bytes, version, applied:["lipsync:<ver>"] } on success. A shot
that genuinely cannot be lip-synced (no detectable face) is an HONEST SOFT-DEGRADE: the job COMPLETES
with { ok: false, detail } (note: `detail`, NOT `error` -- RunPod lifts a top-level `error` key to job
status FAILED, which would fail the whole film), and the lipsync module passes the ORIGINAL clip
through untouched. A GENUINE crash returns { ok: false, error } and lands FAILED so the render fails
loud (vivijure #245).
"""

import copy
import glob
import ipaddress
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

import boto3
import numpy as np
import requests
import runpod
import torch

MUSETALK_DIR = os.environ.get("MUSETALK_DIR", "/app/MuseTalk")
WHISPER_DIR = os.path.join(MUSETALK_DIR, "models", "whisper")


def _ensure_musetalk_path():
    """Put the MuseTalk source checkout on sys.path so the in-process `import musetalk` resolves.

    MuseTalk is a git CHECKOUT baked at MUSETALK_DIR (base.Dockerfile), NOT a pip-installed package, and
    this handler drives it IN-PROCESS. The image launches `python /app/handler.py`, which puts the SCRIPT
    dir (/app) on sys.path[0], NOT the cwd (/app/MuseTalk); nothing else adds it (no PYTHONPATH in the
    image env, no chdir here). Without this the lazy `import musetalk` fails with ModuleNotFoundError
    (#27). Front-insert for priority; idempotent so a re-import does not duplicate the entry."""
    if MUSETALK_DIR not in sys.path:
        sys.path.insert(0, MUSETALK_DIR)


_ensure_musetalk_path()
DOWNLOAD_TIMEOUT = 900
UPLOAD_TIMEOUT = 900

# Inference constants pinned to MuseTalk's scripts.inference CLI defaults (so in-process output matches
# what the old subprocess produced): batch 8, extra face margin 10, jaw parsing, 2/2 audio padding.
BATCH_SIZE = 8
EXTRA_MARGIN = 10
PARSING_MODE = "jaw"
AUDIO_PAD_LEFT = 2
AUDIO_PAD_RIGHT = 2

# #702: minimum fraction of the audio-driven frames a lip-sync must actually emit before we trust it.
# Below this the face was undetectable for most of the clip, so the "synced" output is a stutter far
# shorter than its audio (shot_01 once shipped 3 of 65 frames -> 0.17s). We SoftDegrade to the ORIGINAL
# clip -- honest, full-length, un-synced -- rather than a truncated lie. Mirrors the studio duration gate
# floor (vivijure #697).
LIPSYNC_MIN_FRAME_RATIO = 0.5

# UNet weights, relative to MUSETALK_DIR (inference runs with cwd=MUSETALK_DIR).
UNET = {
    "v15": ("models/musetalkV15/unet.pth", "models/musetalkV15/musetalk.json"),
    "v1": ("models/musetalk/pytorch_model.bin", "models/musetalk/musetalk.json"),
}

R2_ENDPOINT = os.environ.get("R2_ENDPOINT_URL", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "vivijure")

# Warm-worker model cache: version -> loaded pipeline dict. The ~5GB load happens once per worker; every
# subsequent job on that worker reuses it. Keyed by version because the face-parse config differs (v15).
_PIPE = {}


class SoftDegrade(Exception):
    """An honest no-op outcome: the clip genuinely cannot be lip-synced (no detectable face), so the
    module must pass the ORIGINAL clip through unchanged rather than fail the render. The caller turns
    this into a COMPLETED job with {"ok": false, "detail": ...} and NO top-level `error` key, because
    RunPod lifts a top-level `error` to job status FAILED (which would fail the whole film). A GENUINE
    crash is NOT a SoftDegrade: it keeps returning `error` / raising, so the job lands FAILED and the
    render fails loud (vivijure #245)."""


def _r2():
    return boto3.client(
        "s3", endpoint_url=R2_ENDPOINT, region_name="auto",
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
    )


# Optional pin for presigned hosts (e.g. ".r2.cloudflarestorage.com"). Empty = skip host-suffix check.
R2_URL_HOST_SUFFIX = os.environ.get("R2_URL_HOST_SUFFIX", "").strip().lower()


def _url_error(url, what):
    """Refuse non-https / private / link-local / loopback / optional non-R2 host. Returns err str or None.

    Presigned mode otherwise lets any job submitter drive GET/PUT from the GPU worker (SSRF). Resolve
    the hostname and reject blocked address classes; callers must also pass allow_redirects=False so a
    public host cannot bounce into a private one via Location."""
    try:
        p = urlparse(str(url or ""))
    except Exception:  # noqa: BLE001 -- malformed URL is a job error, not a crash
        return f"{what}: malformed URL"
    if p.scheme != "https" or not p.hostname:
        return f"{what}: URL must be https with a hostname"
    host = p.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return f"{what}: URL host is blocked"
    if R2_URL_HOST_SUFFIX:
        suffix = R2_URL_HOST_SUFFIX if R2_URL_HOST_SUFFIX.startswith(".") else f".{R2_URL_HOST_SUFFIX}"
        bare = suffix.lstrip(".")
        if host != bare and not host.endswith(suffix):
            return f"{what}: URL host must end with {R2_URL_HOST_SUFFIX}"
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return f"{what}: URL host does not resolve"
    for _fam, _type, _proto, _canon, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return f"{what}: URL resolves to a blocked address"
    return None


def _get(url, dst):
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=False) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def _probe_dur(path):
    """Media duration in seconds (float), or 0.0 if unknown."""
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
        return float((p.stdout or "").strip() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _speech_end_frame(speech_dur, fps, total_frames):
    """Exclusive frame index where dialogue ends; tail frames hold the source mouth at rest (#67).

    When the dialogue WAV is shorter than the face clip, `_pad_audio_to_video()` pads trailing silence
    so the mux keeps full clip length. MuseTalk still drives one output frame per padded-audio frame;
    whisper features on the silence tail are unstable and produce jittery lip motion after the line
    finishes. Frames at/after this index passthrough the source frame instead of blending generated
    mouth motion."""
    if speech_dur <= 0 or fps <= 0 or total_frames <= 0:
        return total_frames
    return min(total_frames, max(1, int(round(speech_dur * fps))))


def _pad_audio_to_video(audio_path, video_path, work):
    """MuseTalk's output length follows the AUDIO track. When the dialogue is shorter than the face
    clip, MuseTalk emits only the synced (talking) segment and TRUNCATES the shot to the dialogue
    length (a 5s i2v shot synced to a 1.4s line came out 1.4s -- the scatter talking-film clip-drop).
    Pad the audio with trailing silence to the face-clip duration so the synced output keeps the FULL
    clip length: the line is spoken at the head, the mouth rests for the remainder (#67 rest-hold on
    the tail in `_run_musetalk`, not generative lip-sync on near-silent whisper). Returns
    `(audio_path_for_mux, speech_dur)` where `speech_dur` is the original dialogue duration (seconds)
    before any pad; the path is the original if no pad is needed or the pad fails."""
    adur = _probe_dur(audio_path)
    vdur = _probe_dur(video_path)
    if vdur <= 0 or adur <= 0 or adur >= vdur - 0.05:
        return audio_path, adur if adur > 0 else 0.0
    padded = os.path.join(work, "audio_padded.wav")
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", audio_path,
                        "-af", "apad", "-t", f"{vdur:.3f}", padded], check=True)
    except Exception:  # noqa: BLE001 -- pad failure falls back to the original audio
        return audio_path, adur
    return padded, adur


def _pipeline(version):
    """Load MuseTalk's models ONCE per worker and cache them (the ~5GB warm state). Imports MuseTalk's
    internals lazily so an import failure is a job error, not a boot crash. Returns the cached dict."""
    version = version if version in UNET else "v15"
    if version in _PIPE:
        return _PIPE[version]
    from transformers import WhisperModel

    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.utils import load_all_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_path, unet_cfg = UNET[version]
    vae, unet, pe = load_all_model(
        unet_model_path=unet_path, vae_type="sd-vae", unet_config=unet_cfg, device=device)
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)
    weight_dtype = unet.model.dtype
    audio_processor = AudioProcessor(feature_extractor_path=WHISPER_DIR)
    whisper = WhisperModel.from_pretrained(WHISPER_DIR).to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    # v15 takes explicit cheek widths (the CLI defaults); v1 takes none.
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90) if version == "v15" else FaceParsing()
    _PIPE[version] = {
        "device": device, "vae": vae, "unet": unet, "pe": pe,
        "timesteps": torch.tensor([0], device=device), "weight_dtype": weight_dtype,
        "audio_processor": audio_processor, "whisper": whisper, "fp": fp,
    }
    return _PIPE[version]


def _blended_frame_name(written):
    """CONTIGUOUS %08d filename for the blended-frame sequence. ffmpeg`s image2 reader stops at the first
    MISSING index, so an emitted frame MUST be numbered by how many frames we have WRITTEN so far, never by
    the source-loop index: a dropped (no-face / degenerate-bbox) frame must not punch a hole that silently
    truncates the whole encode to its first unbroken run. That hole deterministically cut Night_Shift
    shot_01 to 3 of 65 frames whenever an early frame had no detectable face (vivijure #702)."""
    return f"{written:08d}.png"


def _lipsync_too_short(written, expected, ratio=LIPSYNC_MIN_FRAME_RATIO):
    """True when a lip-sync emitted materially fewer frames than its audio drove -- the face was
    undetectable for most of the clip, so the output is a stutter far shorter than the dialogue. `written`
    = frames actually blended + encoded; `expected` = the audio-driven frame count (vivijure #702)."""
    return expected > 0 and written < expected * ratio


def _run_musetalk(face_path, audio_path, out_path, bbox_shift=0, version="v15"):
    """Lip-sync one face clip against one audio track IN-PROCESS against the warm model cache, and write
    the muxed result to out_path. This mirrors scripts.inference's per-task loop exactly (frame extract ->
    whisper features -> landmark/crop latents -> batched UNet -> blend -> encode + mux), minus the
    per-job model reload. Same signature the callers already use; raises SoftDegrade for a no-face clip
    (callers COMPLETE the job as an honest passthrough) and other exceptions for genuine failures
    (callers return those as FAILED)."""
    import cv2

    from musetalk.utils.blending import get_image
    from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox
    from musetalk.utils.utils import datagen, get_video_fps

    version = version if version in UNET else "v15"
    p = _pipeline(version)
    device, vae, unet, pe = p["device"], p["vae"], p["unet"], p["pe"]
    timesteps, weight_dtype = p["timesteps"], p["weight_dtype"]
    audio_processor, whisper, fp = p["audio_processor"], p["whisper"], p["fp"]
    # v15 uses a fixed bbox_shift of 0; only v1 honours the job-supplied tuning.
    bshift = 0 if version == "v15" else int(bbox_shift or 0)

    work = tempfile.mkdtemp(prefix="ms-infer-")
    frames_dir = os.path.join(work, "frames")
    out_frames_dir = os.path.join(work, "out_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(out_frames_dir, exist_ok=True)
    try:
        # Pad a short dialogue track to the face-clip duration so MuseTalk keeps the full clip length.
        mux_audio_path, speech_dur = _pad_audio_to_video(audio_path, face_path, work)

        # Extract source frames.
        subprocess.run(["ffmpeg", "-v", "fatal", "-y", "-i", face_path, "-start_number", "0",
                        os.path.join(frames_dir, "%08d.png")], check=True)
        input_img_list = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not input_img_list:
            raise RuntimeError("no frames extracted from face clip")
        fps = get_video_fps(face_path)
        speech_end = _speech_end_frame(speech_dur, fps, len(input_img_list))

        # Whisper audio features (padded audio drives one output frame per clip frame).
        with torch.no_grad():
            feats, librosa_length = audio_processor.get_audio_feature(mux_audio_path)
            whisper_chunks = audio_processor.get_whisper_chunk(
                feats, device, weight_dtype, whisper, librosa_length, fps=fps,
                audio_padding_length_left=AUDIO_PAD_LEFT, audio_padding_length_right=AUDIO_PAD_RIGHT)

            # Landmark + crop each frame to a UNet latent. get_landmark_and_bbox prints a bbox_shift
            # hint that AVERAGES over the DETECTED faces (int(sum(average_range_*) / len(average_range_*)));
            # a clip with ZERO detections across every frame leaves those lists empty, so MuseTalk itself
            # raises ZeroDivisionError before it returns. That is the plainest no-face case (scenery that
            # does not even fool the detector, e.g. narration over a landscape) -- an honest soft-degrade,
            # not a crash. Catch ONLY that division; any other error still propagates as a genuine failure.
            try:
                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bshift)
            except ZeroDivisionError:
                raise SoftDegrade("no face detected in clip") from None
            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                if version == "v15":
                    y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])
                crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
                input_latent_list.append(vae.get_latents_for_unet(crop))
            if not input_latent_list:
                raise SoftDegrade("no face detected in clip")

            # Cycle padding so the first/last frames transition smoothly.
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            # Batched UNet inference over the whisper chunks.
            res_frame_list = []
            gen = datagen(whisper_chunks=whisper_chunks, vae_encode_latents=input_latent_list_cycle,
                          batch_size=BATCH_SIZE, delay_frame=0, device=device)
            for whisper_batch, latent_batch in gen:
                audio_feature_batch = pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)
                pred_latents = unet.model(
                    latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                for res_frame in vae.decode_latents(pred_latents):
                    res_frame_list.append(res_frame)

        # Blend each generated mouth back into its source frame. `written` counts frames that landed
        # on a real face bbox: a placeholder / degenerate bbox (a false-positive detection on a faceless
        # clip, e.g. a lighthouse) resizes to a zero area and is dropped here, exactly as upstream does.
        written = 0
        for i, res_frame in enumerate(res_frame_list):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
            if i >= speech_end:
                # #67: padded silence tail -- hold the source mouth at rest, not generative sync.
                cv2.imwrite(os.path.join(out_frames_dir, _blended_frame_name(written)), ori_frame)
                written += 1
                continue
            x1, y1, x2, y2 = bbox
            if version == "v15":
                y2 = min(y2 + EXTRA_MARGIN, ori_frame.shape[0])
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:  # noqa: BLE001 -- a degenerate bbox drops that frame, as upstream does
                continue
            if version == "v15":
                combine = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=PARSING_MODE, fp=fp)
            else:
                combine = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
            # #702: number by frames WRITTEN, not the loop index i. A dropped frame above must not leave
            # a %08d gap -- the image2 encoder below stops at the first missing index, which
            # deterministically truncated shot_01 to 3 of 65 frames (0.17s) on any early no-face frame.
            cv2.imwrite(os.path.join(out_frames_dir, _blended_frame_name(written)), combine)
            written += 1

        # No blended frame landed on a usable face region: the detector false-positived on a faceless
        # clip (input_latent_list was not empty, so the guard above did not fire) but every candidate
        # bbox was a placeholder / degenerate. Honest no-face soft-degrade (a superset of the
        # empty-latent guard), not the confusing post-mux "produced no output mp4".
        if not written:
            raise SoftDegrade("no face detected in clip")
        # #702: the face was detectable for too little of the clip -- the emitted sequence is a stutter far
        # shorter than the audio. Ship the ORIGINAL clip (honest soft-degrade, the lipsync module passes it
        # through), never a truncated lip-sync that the studio duration gate then rejects.
        expected_frames = len(res_frame_list)
        if _lipsync_too_short(written, expected_frames):
            raise SoftDegrade(
                f"lip-sync kept only {written}/{expected_frames} frames "
                "(face undetectable for most of the clip)")

        # Encode the blended frames, then mux the audio back in.
        temp_vid = os.path.join(work, "temp.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "warning", "-r", str(fps), "-f", "image2",
                        "-i", os.path.join(out_frames_dir, "%08d.png"),
                        "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18", temp_vid], check=True)
        # Mux the audio into the CRF-18 video WITHOUT re-encoding the video (-c:v copy). ffmpeg
        # re-encodes by default, and with no codec given it would re-run libx264 at its default
        # (~CRF 23, roughly 2 Mbps at 48fps 720p), silently discarding the CRF-18 first pass above
        # and starving the mouth region MuseTalk just generated (the breathy look an anime 2x upscale
        # then magnifies -- vivijure #584). Stream-copy the video; only the audio is encoded here.
        subprocess.run(["ffmpeg", "-y", "-v", "warning", "-i", mux_audio_path, "-i", temp_vid,
                        "-c:v", "copy", out_path], check=True)
        if not os.path.exists(out_path) or not os.path.getsize(out_path):
            # Inference ran but assembled no usable output (detection too sparse / gapped to mux the
            # %08d.png sequence): an honest no-usable-face soft-degrade, not a hard crash. A genuine
            # ffmpeg failure raises CalledProcessError above (check=True) and stays a FAILED render.
            raise SoftDegrade("no usable face region in clip (produced no lip-synced output)")
        return out_path
    finally:
        shutil.rmtree(work, ignore_errors=True)


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


def _key_error(key, what, prefixes=("renders/",)):
    """Validate a job-supplied R2 key against the render key map BEFORE any bucket I/O. Every key
    this module reads or writes lives inside the studio's render tree (see the module docstring),
    so an absolute key, a `..` segment, a backslash, or an out-of-prefix key is a malformed job.
    Refused as data (this handler reports errors, it does not raise): returns the error string,
    or None when the key is fine."""
    k = str(key or "")
    if not k or k != k.strip() or "\x00" in k:
        return f"{what}: R2 key {k!r} must be a plain relative key under {' or '.join(prefixes)}"
    if k.startswith("/") or "\\" in k or ".." in k.split("/"):
        return f"{what}: R2 key {k!r} must be a plain relative key under {' or '.join(prefixes)}"
    if not k.startswith(tuple(prefixes)):
        return f"{what}: R2 key {k!r} must be a plain relative key under {' or '.join(prefixes)}"
    parts = k.split("/")
    if any(not part for part in parts):
        return f"{what}: R2 key {k!r} must not contain empty path segments"
    if k.endswith("/"):
        return f"{what}: R2 key {k!r} must not end with /"
    return None


def _project_prefix(project):
    """Trusted project segment for shared-bucket tenancy. Mirrors studio finish keys
    (`renders/${project}/...`) -- reject slash/backslash/whitespace so the field cannot widen the prefix."""
    raw = str(project or "")
    p = raw.strip()
    if not p or p != raw or "/" in p or "\\" in p or any(c.isspace() for c in p):
        return None
    return f"renders/{p}/"


def _scoped_key_error(key, what, *, project, prefixes=("renders/",)):
    """Prefix-check plus project tenancy. renders/ and audio/ keys must both sit under
    renders/<project>/ or audio/<project>/ (no flat audio/ exemption -- that was a cross-project read)."""
    err = _key_error(key, what, prefixes=prefixes)
    if err:
        return err
    pref = _project_prefix(project)
    if not pref:
        return f"{what}: project is required for R2 mode"
    p = pref[len("renders/"):].rstrip("/")  # validated project segment
    k = str(key)
    if k.startswith("audio/"):
        audio_pref = f"audio/{p}/"
        if not k.startswith(audio_pref):
            return f"{what}: R2 key must be under {audio_pref}"
        return None
    if not k.startswith(pref):
        return f"{what}: R2 key must be under {pref}"
    return None


def _stamp_sidecar_r2(s3, output_key, output_hash):
    """#583 provenance: write the core-computed param-hash to `<output_key>.hash` AFTER the artifact
    (artifact first, sidecar last -- the only safe order; see the studio CONTRACT.md 3.3.1). The value is
    OPAQUE -- write it verbatim, never recompute it. Best-effort: a failed sidecar only disables reuse
    (the core re-runs), it must NEVER fail a good render. No output_hash (legacy core) -> no sidecar."""
    if not output_hash:
        return
    try:
        s3.put_object(Bucket=R2_BUCKET, Key=f"{output_key}.hash",
                      Body=str(output_hash).encode("utf-8"), ContentType="text/plain")
    except Exception:  # noqa: BLE001 -- provenance is best-effort; a miss = safe re-run, never a failed render
        pass


def _stamp_sidecar_presigned(hash_url, output_hash):
    """Presigned-mode sidecar stamp: the credentialless handler can only write the `.hash` if the core
    presigned a `hash_url` for it. Prod finish uses R2 mode (this is a no-op there); a presigned
    deployment gets provenance once the core presigns hash_url. Same opaque + best-effort contract."""
    if not (hash_url and output_hash):
        return
    err = _url_error(hash_url, "hash_url")
    if err:
        return
    try:
        body = str(output_hash).encode("utf-8")
        requests.put(hash_url, data=body, timeout=UPLOAD_TIMEOUT, allow_redirects=False,
                     headers={"content-type": "text/plain", "content-length": str(len(body))}).raise_for_status()
    except Exception:  # noqa: BLE001 -- best-effort provenance; a miss = safe re-run
        pass


def _lipsync_r2(inp):
    """R2 mode: download clip_key + audio_key, lip-sync, upload output_key in the shared bucket; return
    the new key as `clip_key` so the finish chain carries the lip-synced clip downstream."""
    clip_key = inp.get("clip_key")
    audio_key = inp.get("audio_key")
    project = inp.get("project")
    if not audio_key:
        return {"ok": False, "error": "lipsync needs both clip_key and audio_key"}
    err = (_scoped_key_error(clip_key, "clip_key", project=project)
           # dialogue tracks live under renders/; a staged bed lives under audio/ -- both in-map
           or _scoped_key_error(audio_key, "audio_key", project=project,
                                prefixes=("renders/", "audio/")))
    if err:
        return {"ok": False, "error": err}
    name = clip_key.rsplit("/", 1)[-1]
    output_key = inp.get("output_key") or (
        f"{clip_key.rsplit('.', 1)[0]}_ls.{clip_key.rsplit('.', 1)[1]}" if "." in name else f"{clip_key}_ls")
    err = _scoped_key_error(output_key, "output_key", project=project)
    if err:
        return {"ok": False, "error": err}
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
        _stamp_sidecar_r2(s3, output_key, inp.get("output_hash"))  # #583: sidecar AFTER the artifact
        return {"ok": True, "clip_key": output_key, "bytes": os.path.getsize(dst),
                "version": version, "applied": [f"lipsync:{version}"]}
    except SoftDegrade as e:
        # Honest no-face passthrough: COMPLETES the job ({"ok": false, "detail": ...}, no top-level
        # `error`) so the lipsync module ships the ORIGINAL clip unchanged instead of failing the film.
        return {"ok": False, "detail": str(e)[:500]}
    except Exception as e:  # noqa: BLE001 -- a genuine crash keeps the `error` key -> job FAILED
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
    for u, name in ((video_url, "video_url"), (audio_url, "audio_url"),
                    (output_url, "output_url"), (inp.get("hash_url"), "hash_url")):
        if name == "hash_url" and not u:
            continue  # optional sidecar target
        err = _url_error(u, name)
        if err:
            return {"ok": False, "error": err}
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
            put = requests.put(output_url, data=f, timeout=UPLOAD_TIMEOUT, allow_redirects=False,
                               headers={"content-type": "video/mp4", "content-length": str(size)})
        put.raise_for_status()
        _stamp_sidecar_presigned(inp.get("hash_url"), inp.get("output_hash"))  # #583: sidecar AFTER the artifact
        return {"ok": True, "output_key": output_key, "bytes": size,
                "version": version, "applied": [f"lipsync:{version}"]}
    except SoftDegrade as e:
        # Honest no-face passthrough: COMPLETES the job ({"ok": false, "detail": ...}, no top-level
        # `error`) so the lipsync module ships the ORIGINAL clip unchanged instead of failing the film.
        return {"ok": False, "detail": str(e)[:500]}
    except Exception as e:  # noqa: BLE001 -- a genuine crash keeps the `error` key -> job FAILED
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
