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

import os
import sys
import types


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
    def download_file(self, bucket, key, dst):
        open(dst, "wb").close()

    def upload_file(self, *a, **k):
        pass


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
