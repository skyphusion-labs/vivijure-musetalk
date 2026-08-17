"""Auth-boundary and body-cap coverage for the serve wrapper (vivijure-musetalk#92).

Two defects shipped in this file and neither was guarded, in this repo OR in the
vivijure-audio-upscale reference it was ported from:

  1. `POST /run` answered {"selftest": true} with 200 BEFORE calling token_error(),
     so the door was an UNAUTHENTICATED ORACLE on whatever network it was bound to.
  2. `_body()` read `content-length` with no bound, so a single request could ask the
     process to allocate an arbitrary number of bytes.

The selftest half has a second edge worth stating, because "just answer selftest at the
wrapper" looks harmless: the handler's own {"selftest": true} path is the documented
deploy-verification GPU check (it loads the model and runs a REAL lipsync). Answering it
at this layer returns ok:true on a box with no GPU, a broken model or a missing weight --
a deploy check STRUCTURALLY INCAPABLE OF FAILING. So selftest must be FORWARDED, and that
is asserted here rather than left as a comment (vivijure-upscale#88).

Stdlib only: this module has no heavy deps, so unlike tests/test_handler_routing.py
nothing needs stubbing.

The body cap is exercised against a real server in a SUBPROCESS, not a thread. `_body()`
is a closure inside run_serve() so no unit-level seam reaches it, and run_serve() installs
a SIGTERM handler, which raises ValueError anywhere but the main thread -- a threaded
fixture never comes up, and every test depending on it ERRORS identically whether or not
the fix is present. A subprocess also runs run_serve() in its own main thread, which is
how production runs it.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)

import runpod_http_serve as S  # noqa: E402

TOKEN = "test-token-not-a-real-secret"


def _route(method, path, body, token, expected=TOKEN):
    """Drive route() with a throwaway registry; returns (status, payload)."""
    registry = S.JobRegistry(lambda payload, should_cancel: {"ok": True})
    return S.route(
        method, path, body,
        registry=registry, token=token, expected_token=expected,
        service="test-service",
    )


# ---------------------------------------------------------------- defect 1: auth

@pytest.mark.parametrize("body", [
    {"selftest": True},                 # top-level
    {"input": {"selftest": True}},      # nested under input
    {"selftest": 1},                    # truthy non-bool
])
def test_selftest_is_refused_without_a_token(body):
    """THE DEFECT: this returned 200 with no credential. It must be 401."""
    status, payload = _route("POST", "/run", body, token=None)
    assert status == 401, f"unauthenticated selftest answered {status}: {payload}"
    assert payload.get("error") == "unauthorized"


def test_selftest_is_refused_with_a_WRONG_token():
    status, payload = _route("POST", "/run", {"selftest": True}, token="wrong-token")
    assert status == 401
    assert payload.get("error") == "unauthorized"


def test_ordinary_job_still_refused_without_a_token():
    """Control: proves the 401 above is the auth boundary, not something selftest-specific."""
    status, _ = _route("POST", "/run", {"input": {"project": "p"}}, token=None)
    assert status == 401


def test_unconfigured_token_refuses_503_rather_than_running_open():
    status, payload = _route("POST", "/run", {"selftest": True}, token=None, expected="")
    assert status == 503
    assert "refusing open GPU endpoint" in payload.get("error", "")


def test_health_stays_auth_free():
    """The thing the fix must NOT break: /health is the liveness probe and takes no token."""
    status, payload = _route("GET", "/health", None, token=None)
    assert status == 200
    assert payload["ok"] is True


# ------------------------------------------------- selftest must reach the handler

def test_selftest_is_forwarded_to_the_handler_not_intercepted():
    """A wrapper-answered selftest is a deploy check that cannot fail. It must be submitted."""
    seen = []
    registry = S.JobRegistry(lambda payload, should_cancel: seen.append(payload) or {"ok": True})
    status, payload = S.route(
        "POST", "/run", {"input": {"selftest": True}},
        registry=registry, token=TOKEN, expected_token=TOKEN, service="test-service",
    )
    assert status == 200
    assert "id" in payload, f"selftest was intercepted, not submitted: {payload}"
    assert "selftest" not in payload, "wrapper answered selftest itself"

    deadline = time.time() + 10
    while not seen and time.time() < deadline:
        time.sleep(0.02)
    assert seen == [{"selftest": True}], f"handler never received the selftest job: {seen}"


# ------------------------------------------------------------ defect 2: body cap

def test_body_cap_constant_is_sane():
    assert S.MAX_HTTP_BODY_BYTES == 1_048_576


# ------------------------------------------------- #94: rejected body is not an empty job

def test_oversize_sentinel_is_413_and_does_not_submit():
    """route() used to collapse BODY_TOO_LARGE into (body or {}), i.e. an empty job."""
    seen = []
    registry = S.JobRegistry(lambda payload, should_cancel: seen.append(payload) or {"ok": True})
    status, payload = S.route(
        "POST", "/run", S.BODY_TOO_LARGE,
        registry=registry, token=TOKEN, expected_token=TOKEN, service="test-service",
    )
    assert status == 413, f"oversize sentinel accepted as a job: {status} {payload}"
    assert payload.get("ok") is False
    assert "exceeds" in payload.get("error", "")
    time.sleep(0.05)
    assert seen == [], f"oversize sentinel was submitted: {seen}"


def test_invalid_sentinel_is_400_and_does_not_submit():
    seen = []
    registry = S.JobRegistry(lambda payload, should_cancel: seen.append(payload) or {"ok": True})
    status, payload = S.route(
        "POST", "/run", S.BODY_INVALID,
        registry=registry, token=TOKEN, expected_token=TOKEN, service="test-service",
    )
    assert status == 400, f"invalid sentinel accepted as a job: {status} {payload}"
    assert payload.get("ok") is False
    assert "not valid JSON" in payload.get("error", "")
    time.sleep(0.05)
    assert seen == [], f"invalid sentinel was submitted: {seen}"


def test_oversize_sentinel_without_a_token_is_401_not_413():
    """Auth first: an unauthenticated caller must not learn the cap (#93 ordering)."""
    status, payload = _route("POST", "/run", S.BODY_TOO_LARGE, token=None)
    assert status == 401, f"unauthenticated oversize disclosed the cap: {status} {payload}"
    assert payload.get("error") == "unauthorized"


def test_invalid_sentinel_without_a_token_is_401_not_400():
    status, payload = _route("POST", "/run", S.BODY_INVALID, token=None)
    assert status == 401, f"unauthenticated invalid body skipped auth: {status} {payload}"
    assert payload.get("error") == "unauthorized"


def test_absent_body_is_still_accepted_as_an_empty_job():
    """Control: None (no body) must stay distinct from a rejected body, or /run with
    an empty POST would start returning 413/400 and this fix would have broken the door."""
    seen = []
    registry = S.JobRegistry(lambda payload, should_cancel: seen.append(payload) or {"ok": True})
    status, payload = S.route(
        "POST", "/run", None,
        registry=registry, token=TOKEN, expected_token=TOKEN, service="test-service",
    )
    assert status == 200 and "id" in payload
    deadline = time.time() + 10
    while not seen and time.time() < deadline:
        time.sleep(0.02)
    assert seen == [{}], f"absent body did not become an empty job: {seen}"


SERVER_SRC = '''
import json, sys
sys.path.insert(0, {repo!r})
import runpod_http_serve as S

record = {record!r}

def handler(job):
    with open(record, "a") as fh:
        fh.write(json.dumps(job.get("input")) + "\\n")
    return {{"ok": True}}

S.run_serve(handler, service="test", host="127.0.0.1", port={port})
'''


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Real server in a SUBPROCESS. See the module docstring for why not a thread."""
    tmp = tmp_path_factory.mktemp("serve")
    record = str(tmp / "received.jsonl")
    script = tmp / "server.py"
    port = _free_port()
    script.write_text(SERVER_SRC.format(repo=REPO, record=record, port=port))

    # The server reads its expected token from the environment. Without this the
    # process refuses EVERYTHING with 503 (token not configured), and every live
    # assertion below would be measuring a refusal rather than the path under test.
    env = dict(os.environ, LOCAL_FINISH_TOKEN=TOKEN)
    proc = subprocess.Popen([sys.executable, str(script)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail("server exited early: %s" % (proc.stdout.read() or b"").decode()[:2000])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            break
        except Exception:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("server never came up")

    yield port, record
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _received(record):
    if not os.path.isfile(record):
        return []
    with open(record) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _await_one_more(record, before, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = _received(record)
        if len(got) > before:
            return got
        time.sleep(0.02)
    return _received(record)


def _post(port, raw: bytes, token=TOKEN):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/run", data=raw, method="POST",
        headers={"content-type": "application/json", "authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _post_claimed_length(port, raw: bytes, claimed_length: int, token=TOKEN):
    """POST with a Content-Length that is not the bytes we actually send.

    _body() decides oversize from the header and returns BEFORE rfile.read, so
    a live over-cap test must not ship a 1 MiB+ body: the door closes without
    reading and the client is reset mid-send. Lying about the length is the
    actual decision the cap makes.
    """
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        conn.putrequest("POST", "/run")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Authorization", f"Bearer {token}")
        conn.putheader("Content-Length", str(claimed_length))
        conn.endheaders()
        if raw:
            conn.send(raw)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read() or b"{}")
    finally:
        conn.close()


def test_body_under_the_cap_is_delivered(live_server):
    """POSITIVE CONTROL: without this, the over-cap test could pass against a dead server."""
    port, record = live_server
    before = len(_received(record))
    status, payload = _post(port, json.dumps({"input": {"project": "small"}}).encode())
    assert status == 200 and "id" in payload
    got = _await_one_more(record, before)
    assert len(got) > before, "under-cap job never reached the handler"
    assert got[-1] == {"project": "small"}, f"under-cap body not delivered: {got[-1]}"


def test_body_over_the_cap_is_413_not_an_empty_job(live_server):
    """#94: over-cap used to return 200 + a job id whose payload was {}. Now 413, no job."""
    port, record = live_server
    before = len(_received(record))
    claimed = S.MAX_HTTP_BODY_BYTES + 1
    status, payload = _post_claimed_length(port, b'{"input":{"project":"x"}}', claimed)
    assert status == 413, f"over-cap accepted as a job: {status} {payload}"
    assert payload.get("ok") is False
    assert "exceeds" in payload.get("error", "")
    time.sleep(0.1)
    got = _received(record)
    assert len(got) == before, f"over-cap body was submitted: {got[before:]}"


def test_unparseable_body_is_400_not_an_empty_job(live_server):
    port, record = live_server
    before = len(_received(record))
    status, payload = _post(port, b"this is not json {")
    assert status == 400, f"unparseable body accepted as a job: {status} {payload}"
    assert payload.get("ok") is False
    assert "not valid JSON" in payload.get("error", "")
    time.sleep(0.1)
    got = _received(record)
    assert len(got) == before, f"unparseable body was submitted: {got[before:]}"


def test_oversize_without_a_token_is_401_over_a_real_socket(live_server):
    """Auth still wins over the cap: 413 here would leak the limit to anyone who can POST."""
    port, _ = live_server
    claimed = S.MAX_HTTP_BODY_BYTES + 1
    status, payload = _post_claimed_length(
        port, b'{"input":{"project":"x"}}', claimed, token="wrong-token",
    )
    assert status == 401, f"unauthenticated oversize disclosed the cap: {status} {payload}"
    assert payload.get("error") == "unauthorized"


def test_unauthenticated_selftest_over_a_real_socket(live_server):
    """End-to-end form of defect 1, over the wire rather than through route()."""
    port, _ = live_server
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/run", data=json.dumps({"selftest": True}).encode(),
        method="POST", headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            pytest.fail(f"unauthenticated selftest answered {r.status} over a real socket")
    except urllib.error.HTTPError as e:
        assert e.code == 401
