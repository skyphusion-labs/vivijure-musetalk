"""Unit coverage for the SoftDegrade envelope contract (no GPU needed).

RunPod marks any handler return that carries a top-level `error` key as job status FAILED. The lipsync
module must therefore distinguish an HONEST no-face soft-degrade (the job COMPLETES with
{"ok": false, "detail": ...} so the module passes the ORIGINAL clip through) from a GENUINE crash (the
job lands FAILED with {"ok": false, "error": ...} so the render fails loud, vivijure #245). This asserts
that routing in both handler modes.

The heavy GPU/ML/network deps (torch, boto3, requests, runpod, numpy) import only on the card, so they
are STUBBED here; the routing under test is pure control flow and the tests monkeypatch
_run_musetalk / _get / _r2, so none of the stubs are exercised.
"""

import contextlib
import os
import socket
import sys
import types

import pytest


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


_stub("torch", __version__="0-stub")
_stub("boto3", client=lambda *a, **k: None)
_stub("numpy")
_stub("requests")
# runpod.serverless.start runs at import time (the last line of handler.py); make it a no-op.
_runpod = _stub("runpod")
_runpod.serverless = types.SimpleNamespace(start=lambda *a, **k: None)

# _lipsync_r2 reads these module globals at import for its credential gate.
os.environ.setdefault("R2_ENDPOINT_URL", "https://stub.r2")
os.environ.setdefault("R2_ACCESS_KEY_ID", "stub")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "stub")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler  # noqa: E402


class _FakeS3:
    def __init__(self):
        self.uploaded = []
        self.puts = []          # (Key, Body) for put_object -- captures the .hash sidecar
        self.order = []         # write order, to assert artifact-first / sidecar-last

    def download_file(self, bucket, key, dst):
        open(dst, "wb").close()

    def upload_file(self, src, bucket, key, **k):
        self.uploaded.append(key)
        self.order.append(("artifact", key))

    def put_object(self, Bucket=None, Key=None, Body=None, **k):
        self.puts.append((Key, Body))
        self.order.append(("sidecar", Key))


def _run_ok(face, audio, out, **k):
    with open(out, "wb") as f:
        f.write(b"video-bytes")


def _touch(url, dst):
    with open(dst, "wb") as f:
        f.write(b"x")


def _raise_no_face(*a, **k):
    raise handler.SoftDegrade("no face detected in clip")


def _raise_crash(*a, **k):
    raise RuntimeError("cuda oom")


R2_JOB = {
    "project": "p",
    "clip_key": "renders/p/clips/s.mp4",
    "audio_key": "renders/p/audio/s.wav",
}
# Public https URLs; getaddrinfo is monkeypatched in tests that hit _url_error.
PRESIGNED_JOB = {
    "video_url": "https://bucket.example/v",
    "audio_url": "https://bucket.example/a",
    "output_url": "https://bucket.example/o",
    "output_key": "renders/p/clips/s_ls.mp4",
}


def _public_addrinfo(host, port, *a, **k):
    # 8.8.8.8 is public; TEST-NET / documentation ranges are is_reserved in ipaddress.
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]


@pytest.fixture(autouse=True)
def _allow_presigned_hosts(monkeypatch):
    """Presigned success-path tests need _url_error to accept fixture hosts without real DNS."""
    monkeypatch.setattr(socket, "getaddrinfo", _public_addrinfo)


def test_r2_no_face_completes_with_detail(monkeypatch):
    monkeypatch.setattr(handler, "_r2", lambda: _FakeS3())
    monkeypatch.setattr(handler, "_run_musetalk", _raise_no_face)
    out = handler._lipsync_r2(dict(R2_JOB))
    assert out["ok"] is False
    assert "detail" in out and "error" not in out
    assert "no face" in out["detail"]


def test_r2_crash_keeps_error_key(monkeypatch):
    monkeypatch.setattr(handler, "_r2", lambda: _FakeS3())
    monkeypatch.setattr(handler, "_run_musetalk", _raise_crash)
    out = handler._lipsync_r2(dict(R2_JOB))
    assert out["ok"] is False
    assert "error" in out and "detail" not in out


def test_presigned_no_face_completes_with_detail(monkeypatch):
    monkeypatch.setattr(handler, "_get", _touch)
    monkeypatch.setattr(handler, "_run_musetalk", _raise_no_face)
    out = handler._lipsync_presigned(dict(PRESIGNED_JOB))
    assert out["ok"] is False
    assert "detail" in out and "error" not in out


def test_presigned_crash_keeps_error_key(monkeypatch):
    monkeypatch.setattr(handler, "_get", _touch)
    monkeypatch.setattr(handler, "_run_musetalk", _raise_crash)
    out = handler._lipsync_presigned(dict(PRESIGNED_JOB))
    assert out["ok"] is False
    assert "error" in out and "detail" not in out


def _raise_too_short(*a, **k):
    # #702: the too-short honesty guard fires when the face was undetectable for most of the clip.
    raise handler.SoftDegrade("lip-sync kept only 3/65 frames (face undetectable for most of the clip)")


def test_r2_too_short_completes_with_detail(monkeypatch):
    # A truncated lip-sync (#702) is an HONEST soft-degrade, not a crash: the job COMPLETES with detail so
    # the module ships the ORIGINAL clip, exactly like the no-face path -- never a top-level `error`.
    monkeypatch.setattr(handler, "_r2", lambda: _FakeS3())
    monkeypatch.setattr(handler, "_run_musetalk", _raise_too_short)
    out = handler._lipsync_r2(dict(R2_JOB))
    assert out["ok"] is False
    assert "detail" in out and "error" not in out
    assert "kept only" in out["detail"]


def test_presigned_too_short_completes_with_detail(monkeypatch):
    monkeypatch.setattr(handler, "_get", _touch)
    monkeypatch.setattr(handler, "_run_musetalk", _raise_too_short)
    out = handler._lipsync_presigned(dict(PRESIGNED_JOB))
    assert out["ok"] is False
    assert "detail" in out and "error" not in out


def test_ensure_musetalk_path_inserts_at_front():
    # The in-process handler must put the MuseTalk checkout on sys.path (defect #27). Verify the helper
    # front-inserts MUSETALK_DIR (priority) and is idempotent (no duplicate on re-import).
    saved = list(sys.path)
    try:
        sys.path[:] = [p for p in sys.path if p != handler.MUSETALK_DIR]
        assert handler.MUSETALK_DIR not in sys.path
        handler._ensure_musetalk_path()
        assert sys.path[0] == handler.MUSETALK_DIR
        handler._ensure_musetalk_path()
        assert sys.path.count(handler.MUSETALK_DIR) == 1
    finally:
        sys.path[:] = saved


def _fake_musetalk_raising_zerodiv(monkeypatch):
    """Inject a fake `musetalk` package tree whose get_landmark_and_bbox raises ZeroDivisionError
    (MuseTalk's own 0/0 bbox-shift-hint average over zero detections), plus the other lazy imports
    _run_musetalk pulls. Registered via monkeypatch so pytest unwinds them after the test."""
    def _zdiv(*a, **k):
        raise ZeroDivisionError("division by zero")

    mt = types.ModuleType("musetalk")
    utils = types.ModuleType("musetalk.utils")
    blending = types.ModuleType("musetalk.utils.blending")
    blending.get_image = lambda *a, **k: None
    pre = types.ModuleType("musetalk.utils.preprocessing")
    pre.coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    pre.get_landmark_and_bbox = _zdiv
    u = types.ModuleType("musetalk.utils.utils")
    u.datagen = lambda *a, **k: iter(())
    u.get_video_fps = lambda *a, **k: 25
    mt.utils = utils
    utils.blending = blending
    utils.preprocessing = pre
    utils.utils = u
    for name, mod in [
        ("musetalk", mt), ("musetalk.utils", utils), ("musetalk.utils.blending", blending),
        ("musetalk.utils.preprocessing", pre), ("musetalk.utils.utils", u), ("cv2", types.ModuleType("cv2")),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_zero_detection_detection_raises_softdegrade(monkeypatch, tmp_path):
    # A ZERO-detection clip makes MuseTalk`s get_landmark_and_bbox raise ZeroDivisionError before it
    # returns; _run_musetalk must convert THAT into a SoftDegrade (honest no-face), which the caller
    # then routes to {ok:false, detail} with NO top-level error. Drive _run_musetalk to the detection
    # call with everything upstream stubbed (GPU-free).
    _fake_musetalk_raising_zerodiv(monkeypatch)
    monkeypatch.setattr(sys.modules["torch"], "no_grad", lambda: contextlib.nullcontext(), raising=False)
    monkeypatch.setattr(handler, "_pad_audio_to_video", lambda a, v, w: (a, 0.0))
    monkeypatch.setattr(handler, "_pipeline", lambda version: {
        "device": None, "vae": None, "unet": None, "pe": None, "timesteps": None, "weight_dtype": None,
        "whisper": None, "fp": None,
        "audio_processor": types.SimpleNamespace(
            get_audio_feature=lambda path: (None, 0),
            get_whisper_chunk=lambda *a, **k: []),
    })

    def _fake_run(cmd, *a, **k):
        # the frame-extract ffmpeg writes %08d.png into frames_dir; drop one so the extract check passes
        for tok in cmd:
            if isinstance(tok, str) and tok.endswith("%08d.png"):
                open(tok.replace("%08d", "00000000"), "wb").close()
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(handler.subprocess, "run", _fake_run)

    with pytest.raises(handler.SoftDegrade):
        handler._run_musetalk(str(tmp_path / "face.mp4"), str(tmp_path / "audio.wav"),
                              str(tmp_path / "out.mp4"), version="v15")


def test_zero_detection_routes_to_detail_not_error(monkeypatch):
    # End-to-end envelope: _run_musetalk raising SoftDegrade (the zero-detection outcome above) makes the
    # presigned caller COMPLETE the job as {ok:false, detail}, NO top-level error.
    monkeypatch.setattr(handler, "_get", _touch)
    monkeypatch.setattr(handler, "_run_musetalk", _raise_no_face)
    out = handler._lipsync_presigned(dict(PRESIGNED_JOB))
    assert out["ok"] is False
    assert "detail" in out and "error" not in out


# --- #583 provenance sidecar -------------------------------------------------------------------

def test_r2_stamps_sidecar_after_artifact_when_output_hash_present(monkeypatch):
    s3 = _FakeS3()
    monkeypatch.setattr(handler, "_r2", lambda: s3)
    monkeypatch.setattr(handler, "_run_musetalk", _run_ok)
    out = handler._lipsync_r2({**R2_JOB, "output_key": "renders/p/clips/s_ls.mp4", "output_hash": "deadbeef"})
    assert out["ok"] is True
    # sidecar written to <output_key>.hash with the hash VERBATIM
    assert s3.puts == [("renders/p/clips/s_ls.mp4.hash", b"deadbeef")]
    # artifact FIRST, sidecar LAST (the only safe order)
    assert [kind for kind, _ in s3.order] == ["artifact", "sidecar"]


def test_r2_writes_no_sidecar_without_output_hash(monkeypatch):
    s3 = _FakeS3()
    monkeypatch.setattr(handler, "_r2", lambda: s3)
    monkeypatch.setattr(handler, "_run_musetalk", _run_ok)
    out = handler._lipsync_r2({**R2_JOB, "output_key": "renders/p/clips/s_ls.mp4"})
    assert out["ok"] is True
    assert s3.puts == []  # legacy core (no output_hash) -> no sidecar, safe re-run at the gate


def test_r2_sidecar_write_failure_never_fails_the_render(monkeypatch):
    class _S3Boom(_FakeS3):
        def put_object(self, **k):
            raise RuntimeError("r2 down")
    s3 = _S3Boom()
    monkeypatch.setattr(handler, "_r2", lambda: s3)
    monkeypatch.setattr(handler, "_run_musetalk", _run_ok)
    out = handler._lipsync_r2({**R2_JOB, "output_key": "renders/p/clips/s_ls.mp4", "output_hash": "deadbeef"})
    assert out["ok"] is True and "error" not in out  # artifact is up; a sidecar miss is best-effort


def test_presigned_stamps_sidecar_only_when_hash_url_provided(monkeypatch):
    puts = []
    class _Resp:
        def raise_for_status(self):
            pass
    monkeypatch.setattr(handler, "_get", _touch)
    monkeypatch.setattr(handler, "_run_musetalk", _run_ok)
    def _fake_put(url, data=None, **k):
        puts.append((url, data))
        return _Resp()
    monkeypatch.setattr(handler.requests, "put", _fake_put, raising=False)
    # with hash_url -> sidecar PUT happens (artifact PUT + sidecar PUT = 2)
    handler._lipsync_presigned({**PRESIGNED_JOB, "output_hash": "deadbeef", "hash_url": "https://hash.put"})
    assert ("https://hash.put", b"deadbeef") in puts
    # without hash_url -> no sidecar PUT (only the artifact PUT)
    puts.clear()
    handler._lipsync_presigned({**PRESIGNED_JOB, "output_hash": "deadbeef"})
    assert all(url != "https://hash.put" for url, _ in puts)



# --- #702: the deterministic lip-sync truncation guards (pure, GPU-free) -------------------------------
# Night_Shift shot_01 shipped a 3-of-65-frame clip (0.17s for a 4s shot), deterministically, whenever an
# early source frame had no detectable face. Root cause: blended output PNGs were named by the source LOOP
# INDEX, so a dropped (degenerate-bbox) frame left a %08d hole; ffmpeg`s image2 reader stops at the first
# missing index, truncating the clip to its first unbroken run. These cover the two pure guards that fix it.

def _emit_names_like_the_loop(keep_flags):
    """Reproduce the blend loop`s output-naming for a sequence of keep/drop decisions, using the FIXED
    contiguous counter. A dropped frame (keep=False) is `continue`d exactly as the loop does."""
    names = []
    written = 0
    for keep in keep_flags:
        if not keep:
            continue
        names.append(handler._blended_frame_name(written))
        written += 1
    return names


def test_frame_names_are_contiguous_even_when_frames_drop():
    # Frame index 3 drops (no face) -- the historical shot_01 signature.
    names = _emit_names_like_the_loop([True, True, True, False, True, True])
    assert names == ["00000000.png", "00000001.png", "00000002.png", "00000003.png", "00000004.png"]
    indices = [int(n.split(".")[0]) for n in names]
    assert indices == list(range(len(indices)))  # no hole -> ffmpeg reads every emitted frame


def test_old_loop_index_scheme_would_have_gapped():
    # Regression guard: naming by the loop index i reintroduces the hole ffmpeg truncates on.
    keep = [True, True, True, False, True, True]
    old_scheme = [f"{i:08d}.png" for i, k in enumerate(keep) if k]
    assert "00000003.png" not in old_scheme  # gap at 3 -> ffmpeg stops after 00000002.png (3 frames)
    assert old_scheme != _emit_names_like_the_loop(keep)


def test_no_drops_is_a_full_contiguous_sequence():
    names = _emit_names_like_the_loop([True] * 64)
    assert len(names) == 64 and names[-1] == "00000063.png"


def test_too_short_trips_on_the_shot_01_case():
    assert handler._lipsync_too_short(3, 65) is True     # 3 of 65 -> degrade to the original clip


def test_too_short_passes_full_and_transient_misses():
    assert handler._lipsync_too_short(64, 65) is False   # full-length sync
    assert handler._lipsync_too_short(60, 65) is False   # 92% kept -- a few transient misses


def test_too_short_floor_is_inclusive():
    assert handler._lipsync_too_short(50, 100) is False   # exactly 50% is trusted
    assert handler._lipsync_too_short(49, 100) is True    # strictly below degrades


def test_too_short_never_trips_on_zero_expected():
    assert handler._lipsync_too_short(0, 0) is False


# --- #67: silence-pad tail rest-hold (pure, GPU-free) ----------------------------------------


def test_speech_end_frame_maps_dialogue_duration_to_frame_index():
    # 1.4s dialogue @ 16fps -> frame 22 is first rest-hold (0..21 synced, 22+ passthrough).
    assert handler._speech_end_frame(1.4, 16, 81) == 22


def test_speech_end_frame_clamps_to_clip_length():
    assert handler._speech_end_frame(10.0, 16, 81) == 81


def test_speech_end_frame_unknown_duration_syncs_whole_clip():
    assert handler._speech_end_frame(0.0, 16, 81) == 81


def test_speech_end_frame_keeps_at_least_one_synced_frame():
    assert handler._speech_end_frame(0.01, 16, 81) == 1


def test_pad_audio_tuple_returns_speech_dur_before_pad(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    video = tmp_path / "v.mp4"
    audio.touch()
    video.touch()
    monkeypatch.setattr(handler, "_probe_dur", lambda p: 1.4 if p == str(audio) else 5.0)
    path, speech_dur = handler._pad_audio_to_video(str(audio), str(video), str(tmp_path / "work"))
    assert speech_dur == 1.4
    assert path == str(audio) or path.endswith("audio_padded.wav")


def test_pad_audio_no_pad_when_dialogue_fills_clip(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    video = tmp_path / "v.mp4"
    audio.touch()
    video.touch()
    monkeypatch.setattr(handler, "_probe_dur", lambda p: 4.98 if p == str(audio) else 5.0)
    path, speech_dur = handler._pad_audio_to_video(str(audio), str(video), str(tmp_path / "work"))
    assert path == str(audio)
    assert speech_dur == 4.98


# --- Presigned URL SSRF gate -----------------------------------------------------------------


def test_url_error_rejects_http_and_private(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))])
    assert handler._url_error("http://evil.example/x", "video_url")
    assert handler._url_error("https://127.0.0.1/x", "video_url")
    assert "blocked" in handler._url_error("https://loop.example/x", "video_url")


def test_url_error_accepts_public_https(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _public_addrinfo)
    assert handler._url_error("https://bucket.example/obj", "video_url") is None


def test_url_error_host_suffix_pin(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _public_addrinfo)
    monkeypatch.setattr(handler, "R2_URL_HOST_SUFFIX", ".r2.cloudflarestorage.com")
    assert handler._url_error("https://evil.example/x", "video_url")
    assert handler._url_error(
        "https://acct.r2.cloudflarestorage.com/obj", "video_url") is None


def test_presigned_rejects_bad_url_before_get(monkeypatch):
    called = {"get": 0}

    def boom(*a, **k):
        called["get"] += 1
        raise AssertionError("_get must not run for rejected URLs")

    monkeypatch.setattr(handler, "_get", boom)
    monkeypatch.setattr(handler, "_run_musetalk", _run_ok)
    out = handler._lipsync_presigned({
        "video_url": "http://169.254.169.254/latest",
        "audio_url": "https://bucket.example/a",
        "output_url": "https://bucket.example/o",
    })
    assert out["ok"] is False and "error" in out
    assert called["get"] == 0


def test_r2_rejects_cross_project_before_io(monkeypatch):
    class Boom:
        def download_file(self, *a, **k):
            raise AssertionError("must not touch R2 for rejected keys")

        def upload_file(self, *a, **k):
            raise AssertionError("must not touch R2 for rejected keys")

    monkeypatch.setattr(handler, "_r2", lambda: Boom())
    out = handler._lipsync_r2({
        "project": "attacker",
        "clip_key": "renders/victim/clips/s.mp4",
        "audio_key": "renders/victim/audio/s.wav",
    })
    assert out["ok"] is False
    assert "must be under renders/attacker/" in out["error"]


def test_r2_rejects_missing_project(monkeypatch):
    monkeypatch.setattr(handler, "_r2", lambda: None)
    out = handler._lipsync_r2({
        "clip_key": "renders/p/clips/s.mp4",
        "audio_key": "renders/p/audio/s.wav",
    })
    assert out["ok"] is False
    assert "project is required" in out["error"]


def test_r2_rejects_flat_audio_prefix(monkeypatch):
    monkeypatch.setattr(handler, "_r2", lambda: None)
    out = handler._lipsync_r2({
        "project": "neon",
        "clip_key": "renders/neon/clips/s.mp4",
        "audio_key": "audio/uuid.wav",  # flat staging -- no project segment
    })
    assert out["ok"] is False
    assert "must be under audio/neon/" in out["error"]


def test_r2_accepts_project_scoped_audio_prefix():
    assert handler._scoped_key_error(
        "audio/neon/s.wav", "audio_key", project="neon",
        prefixes=("renders/", "audio/")) is None


def test_key_error_rejects_empty_segments_and_trailing_slash():
    assert handler._key_error("renders/p//clips/s.mp4", "clip_key") is not None
    assert handler._key_error("renders/p/clips/", "clip_key") is not None
    assert handler._key_error("renders/p/clips/s.mp4\x00", "clip_key") is not None
