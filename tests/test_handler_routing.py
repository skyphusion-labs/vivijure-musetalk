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


R2_JOB = {"clip_key": "renders/p/clips/s.mp4", "audio_key": "renders/p/audio/s.wav"}
PRESIGNED_JOB = {
    "video_url": "u", "audio_url": "u", "output_url": "u", "output_key": "renders/p/clips/s_ls.mp4",
}


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
    monkeypatch.setattr(handler, "_pad_audio_to_video", lambda a, v, w: a)
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
