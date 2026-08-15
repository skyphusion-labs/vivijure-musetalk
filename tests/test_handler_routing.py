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

import ast
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
# urllib3 backs DNS-pinned presigned fetches; stub before handler import (CI installs pytest only).
_urllib3_exc = types.SimpleNamespace(HTTPError=Exception)
_stub("urllib3", Timeout=lambda **k: object(), HTTPSConnectionPool=lambda *a, **k: None,
       exceptions=_urllib3_exc)
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


def _touch(url, dst, deadline=None):
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
    monkeypatch.setattr(handler, "_pad_audio_to_video", lambda a, v, w, **k: (a, 0.0))
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
    monkeypatch.setattr(handler, "_get", _touch)
    monkeypatch.setattr(handler, "_run_musetalk", _run_ok)

    def _fake_pinned_put(url, body, *, headers):
        puts.append((url, body))

    monkeypatch.setattr(handler, "_pinned_put", _fake_pinned_put)
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


def test_parse_speech_end_sec_uses_first_trailing_silence():
    log = "[silencedetect @ 0x0] silence_start: 1.420\n"
    assert abs(handler._parse_speech_end_sec(log, 5.0) - 1.48) < 0.01


def test_parse_speech_end_sec_falls_back_to_file_dur_without_silence():
    assert handler._parse_speech_end_sec("", 1.42) == 1.42
    assert handler._parse_speech_end_sec("[silencedetect] silence_start: 0.01\n", 5.0) == 5.0


def test_parse_speech_end_sec_ignores_leading_silence_at_zero():
    log = "[silencedetect] silence_start: 0.0\n[silencedetect] silence_end: 0.08\n[silencedetect] silence_start: 1.35\n"
    assert abs(handler._parse_speech_end_sec(log, 5.0) - 1.41) < 0.01


def test_pad_audio_tuple_returns_speech_end_not_container_dur(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    video = tmp_path / "v.mp4"
    audio.touch()
    video.touch()
    monkeypatch.setattr(handler, "_probe_dur", lambda p, **k: 4.98 if p == str(audio) else 5.0)
    monkeypatch.setattr(handler, "_probe_speech_end_sec", lambda p, **k: 1.42)
    path, speech_end = handler._pad_audio_to_video(str(audio), str(video), str(tmp_path / "work"))
    assert path == str(audio)
    assert speech_end == 1.42


def test_pad_audio_tuple_returns_speech_dur_before_pad(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    video = tmp_path / "v.mp4"
    audio.touch()
    video.touch()
    monkeypatch.setattr(handler, "_probe_dur", lambda p, **k: 1.4 if p == str(audio) else 5.0)
    monkeypatch.setattr(handler, "_probe_speech_end_sec", lambda p, **k: 1.4)
    path, speech_end = handler._pad_audio_to_video(str(audio), str(video), str(tmp_path / "work"))
    assert speech_end == 1.4
    assert path == str(audio) or path.endswith("audio_padded.wav")


def test_pad_audio_no_pad_when_dialogue_fills_clip(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    video = tmp_path / "v.mp4"
    audio.touch()
    video.touch()
    monkeypatch.setattr(handler, "_probe_dur", lambda p, **k: 4.98 if p == str(audio) else 5.0)
    monkeypatch.setattr(handler, "_probe_speech_end_sec", lambda p, **k: 1.42)
    path, speech_end = handler._pad_audio_to_video(str(audio), str(video), str(tmp_path / "work"))
    assert path == str(audio)
    assert speech_end == 1.42


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


def test_pinned_get_connects_to_validated_ip(monkeypatch, tmp_path):
    """#69: fetch uses the IP from _resolve_pinned_ip (DNS pinning), not a second lookup."""
    seen = {}

    class _FakeResp:
        status = 200

        @staticmethod
        def stream(_chunk):
            return [b"pinned-bytes"]

        @staticmethod
        def release_conn():
            pass

    class _FakePool:
        def request(self, method, path, headers=None, **_kw):
            seen["method"] = method
            seen["path"] = path
            seen["host"] = headers.get("Host")
            return _FakeResp()

    monkeypatch.setattr(socket, "getaddrinfo", _public_addrinfo)
    monkeypatch.setattr(handler, "_pinned_pool",
                    lambda host, ip, port, **k: (seen.update({"ip": ip, "read_timeout": k.get("read_timeout")})
                                                 or _FakePool()))
    dst = tmp_path / "out.bin"
    handler._pinned_get("https://bucket.example/obj", str(dst))
    assert dst.read_bytes() == b"pinned-bytes"
    assert seen["ip"] == "8.8.8.8"
    assert seen["host"] == "bucket.example"
    assert seen["method"] == "GET"


def test_presigned_rejects_ssrf_hash_url_before_put(monkeypatch):
    puts = {"n": 0}

    def fake_pinned_put(url, body, *, headers):
        puts["n"] += 1
        raise AssertionError("PUT must not run for rejected hash_url")

    monkeypatch.setattr(handler, "_pinned_put", fake_pinned_put)
    monkeypatch.setattr(handler, "_get", lambda *a, **k: None)
    monkeypatch.setattr(handler, "_run_musetalk", _run_ok)
    out = handler._lipsync_presigned({
        **PRESIGNED_JOB,
        "output_hash": "deadbeef",
        "hash_url": "http://169.254.169.254/latest",
    })
    assert out["ok"] is False and "error" in out
    assert puts["n"] == 0


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


# --- #98: the per-invocation wall-clock guard ---------------------------------------------------
# The compute path had NO wall-clock bound of any kind before this: 0 of 6 subprocess.run sites
# carried a timeout=, and the only timeout= in the file was the HTTP read on the presigned download.
# For finish-lipsync the studio phase ceiling (5400s) was the SOLE backstop, and that ceiling fails
# the whole PHASE rather than degrading one step (vivijure-core#182, vivijure-musetalk#98).
#
# A test that only asserts a fast job still succeeds passes identically with and without the guard,
# so it cannot see this change at all. The tests that CAN see it are the ones where the guard FIRES,
# where it is asked NOT to fire, and where an existing broad except could silently swallow it.

REFERENCE_EXECUTION_TIMEOUT_S = 600   # deploy.sh EXECUTION_TIMEOUT_MS default 600000
PHASE_HARD_DEADLINE_S = 5400          # vivijure-core film-model.ts PHASE_HARD_DEADLINE_SECONDS
FINISH_STEP_MAX_ATTEMPTS = 3          # vivijure-core film-model.ts
RUNPOD_COLD_GRACE_S = 900             # vivijure-cf finish-lipsync RUNPOD_COLD_GRACE_MS


def _handler_source():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "handler.py")) as f:
        return f.read()


def test_exactly_one_subprocess_run_site_and_it_is_guarded():
    # Read the SHIPPED SOURCE, not a stub: this is the coverage denominator and it must not depend on
    # which code path a test happens to drive. Every compute subprocess goes through _run_guarded, so
    # the file may contain exactly ONE subprocess.run call and it must carry a timeout.
    tree = ast.parse(_handler_source())
    sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess"]
    # CONTROL, and it is the point of writing it this way: a matcher that finds NOTHING would satisfy
    # the timeout assertion below vacuously. Assert the count first so a zero reddens instead of
    # reading as a clean pass (this issue was once measured with a matcher that could only return 0).
    assert len(sites) == 1, (
        f"expected exactly 1 subprocess.run site (all compute goes through _run_guarded), "
        f"found {len(sites)}")
    kwargs = [k.arg for k in sites[0].keywords]
    assert "timeout" in kwargs, f"the single subprocess.run site must carry timeout=, got {kwargs}"


def test_every_compute_subprocess_site_goes_through_the_guard():
    # The denominator, stated as a number: 6 of 6 compute subprocess sites (ffprobe duration,
    # silencedetect, audio pad, frame extract, encode, mux) are inside the invocation budget.
    tree = ast.parse(_handler_source())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_run_guarded"]
    assert len(calls) == 6, f"expected 6 guarded compute subprocess sites, found {len(calls)}"


def test_default_budget_fits_under_the_platform_kill_and_the_phase_ceiling():
    # The arithmetic that chose 540, made executable so a later bump cannot quietly break either
    # bound. Asserts the DEFAULT (CI does not set MAX_INVOCATION_SECONDS).
    g = handler.MAX_INVOCATION_SECONDS
    assert g < REFERENCE_EXECUTION_TIMEOUT_S, (
        f"a guard at or above the reference deployment execution timeout "
        f"({REFERENCE_EXECUTION_TIMEOUT_S}s) can never fire: the platform kills the worker first, "
        f"and a platform kill fails the render instead of degrading it. Got {g}s.")
    worst = FINISH_STEP_MAX_ATTEMPTS * g + RUNPOD_COLD_GRACE_S + FINISH_STEP_MAX_ATTEMPTS * 60
    assert worst < PHASE_HARD_DEADLINE_S, (
        f"a retry moves attempts and not the progress index, so the worst case is "
        f"{FINISH_STEP_MAX_ATTEMPTS} * {g}s of guard plus cold start plus queue wait = {worst}s, "
        f"which must stay under the {PHASE_HARD_DEADLINE_S}s phase ceiling.")


def test_deadline_reason_names_the_guard_and_the_elapsed_seconds():
    dl = handler._Deadline()
    reason = dl.reason("unet-inference")
    assert "MAX_INVOCATION_SECONDS" in reason      # names the guard, so an operator can find the knob
    assert "unet-inference" in reason              # names the stage that ran out
    assert "after" in reason                       # reports elapsed, not just the fact of expiry
    # The studio truncates this to 120 characters; everything load-bearing has to survive that.
    assert len(reason) <= 120, f"reason is {len(reason)} chars and would be truncated: {reason}"


def test_check_passes_while_budget_remains_and_raises_once_it_is_spent(monkeypatch):
    # Both directions on the same instrument. A guard that only ever raises would pass a one-sided
    # test while degrading every job in production.
    clock = {"t": 100.0}
    monkeypatch.setattr(handler, "time", types.SimpleNamespace(monotonic=lambda: clock["t"]))
    dl = handler._Deadline(seconds=60)
    dl.check("blend")                 # 0s spent of 60: must NOT raise
    clock["t"] += 59.0
    dl.check("blend")                 # 59s spent of 60: must still NOT raise
    clock["t"] += 2.0
    with pytest.raises(handler.SoftDegrade):
        dl.check("blend")             # 61s spent of 60: must raise


def _guard_expiry(*a, **k):
    """Raise exactly what an expired budget raises, built from the real class, not a hand-written string."""
    raise handler.SoftDegrade(handler._Deadline().reason("unet-inference"))


def test_guard_fires_inside_the_pipeline_on_a_slow_stage(monkeypatch, tmp_path):
    # THE DISCRIMINATING TEST. Drive the REAL _run_musetalk with a stubbed slow step (the frame
    # extract returns, but the clock has jumped past the budget), and assert the invocation ends as
    # an honest degrade instead of running on. Before this change there was nothing to end it.
    _fake_musetalk_raising_zerodiv(monkeypatch)
    monkeypatch.setattr(sys.modules["torch"], "no_grad", lambda: contextlib.nullcontext(), raising=False)
    clock = {"t": 5000.0}
    monkeypatch.setattr(handler, "time", types.SimpleNamespace(monotonic=lambda: clock["t"]))
    monkeypatch.setattr(handler, "_pad_audio_to_video", lambda a, v, w, **k: (a, 0.0))
    monkeypatch.setattr(handler, "_pipeline", lambda version: {
        "device": None, "vae": None, "unet": None, "pe": None, "timesteps": None, "weight_dtype": None,
        "whisper": None, "fp": None,
        "audio_processor": types.SimpleNamespace(
            get_audio_feature=lambda path: (None, 0),
            get_whisper_chunk=lambda *a, **k: []),
    })
    timeouts = []

    def _slow_run(cmd, *a, **k):
        timeouts.append(k.get("timeout"))
        for tok in cmd:
            if isinstance(tok, str) and tok.endswith("%08d.png"):
                open(tok.replace("%08d", "00000000"), "wb").close()
        clock["t"] += 10000.0   # the stubbed slow step: this ffmpeg outlived the whole budget
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(handler.subprocess, "run", _slow_run)

    with pytest.raises(handler.SoftDegrade) as exc:
        handler._run_musetalk(str(tmp_path / "face.mp4"), str(tmp_path / "audio.wav"),
                              str(tmp_path / "out.mp4"), version="v15")
    msg = str(exc.value)
    assert "MAX_INVOCATION_SECONDS" in msg
    assert "whisper-features" in msg      # the stage AFTER the slow step is where it stopped
    assert "after 10000.0s" in msg        # the elapsed seconds are real, not a fixed string
    # and the subprocess that did run was itself bounded by the remaining budget, never unbounded
    assert timeouts and all(t is not None and t > 0 for t in timeouts), timeouts


def test_expiry_reaches_the_r2_door_as_detail_and_never_as_error(monkeypatch):
    # What the DOOR emits. {ok:false, detail} is the shape vivijure-cf recovers at
    # modules/finish-lipsync/src/index.ts:331 (on main today): it passes the ORIGINAL clip through,
    # tags applied with passthrough:backend-soft-degrade and puts this reason in degraded. An
    #  key instead would book a FAILED envelope and log outcome:failed for a job that
    # degraded honestly; a RAISE would leave no structured output and fail the whole film.
    monkeypatch.setattr(handler, "_r2", lambda: _FakeS3())
    monkeypatch.setattr(handler, "_run_musetalk", _guard_expiry)
    out = handler._lipsync_r2(dict(R2_JOB))
    assert out["ok"] is False
    assert "detail" in out and "error" not in out
    assert "applied" not in out          # NO tag from the door: cf builds the honest passthrough tag
    assert "MAX_INVOCATION_SECONDS" in out["detail"]
    assert len(out["detail"]) <= 120     # cf truncates at 120; the reason must survive whole


def test_expiry_reaches_the_presigned_door_as_detail_and_never_as_error(monkeypatch):
    monkeypatch.setattr(handler, "_get", _touch)
    monkeypatch.setattr(handler, "_run_musetalk", _guard_expiry)
    out = handler._lipsync_presigned(dict(PRESIGNED_JOB))
    assert out["ok"] is False
    assert "detail" in out and "error" not in out
    assert "applied" not in out
    assert "MAX_INVOCATION_SECONDS" in out["detail"]


def test_expiry_never_escapes_the_door_as_an_exception(monkeypatch):
    # The single most important property: an escaping exception is booked FAILED with no structured
    # output, which vivijure-core classifies as deterministic and fails the ENTIRE render. Assert the
    # door RETURNS on every mode rather than propagating.
    monkeypatch.setattr(handler, "_r2", lambda: _FakeS3())
    monkeypatch.setattr(handler, "_get", _touch)
    monkeypatch.setattr(handler, "_run_musetalk", _guard_expiry)
    assert isinstance(handler.handler({"input": dict(R2_JOB)}), dict)
    assert isinstance(handler.handler({"input": dict(PRESIGNED_JOB)}), dict)


def test_a_normal_job_does_not_trip_the_guard(monkeypatch):
    # The guard must be invisible on the path it is not for. Without this, a guard that fired on
    # everything would still pass every assertion above.
    monkeypatch.setattr(handler, "_r2", lambda: _FakeS3())
    monkeypatch.setattr(handler, "_run_musetalk", _run_ok)
    out = handler._lipsync_r2(dict(R2_JOB))
    assert out["ok"] is True
    assert out["applied"] == ["lipsync:v15"]
    assert "detail" not in out and "error" not in out


def _timeout_run(cmd, *a, **k):
    raise handler.subprocess.TimeoutExpired(cmd, k.get("timeout", 1))


def test_expiry_is_not_swallowed_by_the_duration_fallback(monkeypatch, tmp_path):
    # _probe_dur ends in a broad except Exception that returns 0.0. If the guard were caught there,
    # an expiry would read as an unknown duration and the invocation would carry on unbounded: a
    # guard that is present, tested, and silently inert on the exact path it exists for.
    monkeypatch.setattr(handler.subprocess, "run", _timeout_run)
    with pytest.raises(handler.SoftDegrade):
        handler._probe_dur(str(tmp_path / "a.wav"), deadline=handler._Deadline(seconds=600))


def test_expiry_is_not_swallowed_by_the_audio_pad_fallback(monkeypatch, tmp_path):
    # Same hazard at the pad, whose broad except falls back to the unpadded audio.
    monkeypatch.setattr(handler, "_probe_dur", lambda p, **k: 1.4 if p.endswith("a.wav") else 5.0)
    monkeypatch.setattr(handler, "_probe_speech_end_sec", lambda p, **k: 1.4)
    monkeypatch.setattr(handler.subprocess, "run", _timeout_run)
    with pytest.raises(handler.SoftDegrade):
        handler._pad_audio_to_video(str(tmp_path / "a.wav"), str(tmp_path / "v.mp4"),
                                    str(tmp_path), deadline=handler._Deadline(seconds=600))
