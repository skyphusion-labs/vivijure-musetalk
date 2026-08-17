# Changelog -- vivijure-musetalk

The image ships as a git-tag-driven release (`v<X.Y.Z>`; see `RELEASES.md`). Each tag builds the
consumer image. This file records the why behind each release; the tag is the version of record.

## Unreleased

- **fix(lipsync): passthrough the source frame on a degenerate bbox (#40).** A placeholder or
  zero-area face box used to `continue` out of the blend loop, so a clip that still cleared the
  honesty floor shipped shorter than its audio. The loop now writes the untouched source frame at
  that index (exact audio length, zero dropped frames) and only blends a generated mouth onto a
  usable box. Silence-tail rest-holds (#67) are unchanged. The floor now counts blended mouths
  against the speech-window attempt count (blended + passthrough): a clip that never found a face
  still SoftDegrades, and a 6/64-face clip still cannot ship as a success. Handler + tests only;
  no weight or base change.

- **fix(serve): an oversize or unparseable POST /run is no longer accepted as an empty job (#94).**
  `_body()` answered `None` for no body, a body past the 1 MiB cap, and a body that would not
  parse, and `/run` then did `(body or {}).get("input", body or {})`, so all three were accepted
  with `200` and a job id. The caller got a success shape for a request that was never honoured,
  and the job failed later naming a missing field rather than the body. Now `413` and `400`
  respectively, checked AFTER authentication so an unauthenticated caller still gets `401` and
  learns nothing about the cap. Ported from vivijure-blender.

- **fix(lipsync): a per-invocation wall-clock guard on the compute path (#98).** The handler had no
  wall-clock bound of any kind: 0 of 6 `subprocess.run` sites carried a `timeout=` (the single
  `timeout=` in the file was the HTTP read on the presigned download), and there was no
  `monotonic`, no `signal.alarm`, no clock of any sort in the non-test Python. Of the four finish
  doors this was the one where the studio phase ceiling (`PHASE_HARD_DEADLINE_SECONDS`, 5400s) was
  the SOLE backstop, and that ceiling fails the whole PHASE rather than degrading one step, so one
  unbounded shot took a correctly-running film down with it (vivijure-core#182).
  ONE budget is now established per invocation and threaded through every stage: 6 of 6 subprocess
  sites go through a single guarded entry point that spends the remaining budget as the child
  timeout, and 6 in-process stages (model load, whisper features, face detection, latent encode,
  UNet inference, blending) check it, including inside the unbounded per-frame loops. On expiry the
  job COMPLETES as the existing honest soft-degrade (`{ok:false, detail}`, no top-level `error`),
  naming the guard and the elapsed seconds, so the studio passes the ORIGINAL clip through, tags it
  `passthrough:backend-soft-degrade` and records the reason. It never raises: an escaping exception
  would be booked FAILED with no structured output and fail the entire render.
  Default 540s, env-overridable via `MAX_INVOCATION_SECONDS`: below the 600s reference deployment
  execution timeout (so the honest degrade wins the race against the platform kill) and 3 * 540 plus
  cold start plus queue wait stays at half the 5400s phase ceiling. Handler + tests + docs only; no
  weight or base change.

## v1.0.6

- **fix(hub): align the Hub listing GPU pools and disk with the production endpoint (#79).** The
  listing advertised `BLACKWELL_180,HOPPER_141` and explicitly negated the three RTX PRO 6000 cards
  by name. Those cards ARE the `BLACKWELL_96` pool, which is the pool production endpoint
  `zw6pt4lymf69pk` actually runs this worker on, so the listing excluded the one configuration we
  prove daily and left a Hub deployer on B200 or H200 class hardware at roughly two to three times
  the hourly cost for the same job. `gpuIds` is now `BLACKWELL_96,HOPPER_141,BLACKWELL_180`
  (production pool first, larger pools kept as availability fallbacks; no unproven pool added), and
  `tests.json` runs the Hub smoke on `NVIDIA RTX PRO 6000 Blackwell Server Edition`, the card
  production runs on, so a green Hub test carries the same meaning our own endpoint carries.
  `containerDiskInGb` stays 40, matching production. `.runpod/README.md` records the provenance and
  the repin rule. Endpoint config, GPU pool membership, and image size were read live (read-only).
- **Docs and listing metadata only.** The tag still bakes a consumer image (`build-image.yml` fires
  on `v*` tags), and `:1.0.6` is functionally identical to `:1.0.5`. Production stays pinned to
  `:1.0.5` on purpose; **no repin**.

## v1.0.5

- **fix(lipsync): silencedetect speech boundary + freeze last synced frame (#67, PR #76).** v1.0.3
  rest-hold keyed off ffprobe file duration, so a dialogue WAV already padded to clip length (or with
  trailing silence in the container) still ran generative MuseTalk on the near-silent tail. The handler
  now detects spoken-content end via ffmpeg `silencedetect`, rest-holds from that frame, and freezes the
  last blended mouth (not the raw i2v source) through the silence pad. Handler-only release.

## v1.0.4

- **fix(security): DNS-pin presigned fetches (#69, K3 closeout).** `_pinned_get` / `_pinned_put`
  connect to the IP validated by `_url_error` with correct SNI, closing the DNS-rebinding TOCTOU
  on presigned-mode GET/PUT. Handler-only; no weight/base change.

## v1.0.3

- **fix(lipsync): rest-hold source frames on silence-pad tail (#67, PR #68).** Padded trailing
  silence kept MuseTalk generating unstable mouth motion after dialogue ended. `_pad_audio_to_video()`
  now returns speech duration; frames at/after the speech-end index passthrough the source frame (mouth
  at rest) while the full padded audio track still muxes to the face-clip duration. Handler-only
  release; base image unchanged.

## v1.0.2

- **fix(security): project-scoped R2 + SSRF gate (#63, #64, #65, #66).** Presigned URL validation,
  `project` scope required for `renders/<project>/` and `audio/<project>/`, allowlist sync hardened,
  host builds moved to `ubuntu-latest`, and the adversarial security audit workflow added.
  Handler-only; base unchanged. Image `:1.0.2`.
  (Backfilled 2026-07-25 from the v1.0.2 GitHub release; the row was missing from this file.)

## v1.0.1

- **docs(hub): RunPod Hub publish surface (musetalk#57).** `.runpod/hub.json` + `tests.json`
  (`{"selftest": true}`), `.runpod/README.md` with the R2 env names (`R2_ENDPOINT_URL`),
  `THIRD_PARTY_MODELS.md`, and the Hub badge. Docs-only patch cut so Hub, which indexes releases and
  not commits, could index a release tree containing `.runpod/`. No handler or image-recipe change.
  (Backfilled 2026-07-25: this entry sat under Unreleased, but `git tag --contains` puts the commit
  in v1.0.1 through v1.0.5, so it shipped in v1.0.1.)

## v1.0.0

- **First stable release of the MuseTalk lip-sync finish module.** The lip-sync satellite in the
  Vivijure constellation, output-verified end-to-end for Studio v1.0.0 (finish-lipsync: the MuseTalk
  `_ls` artifact is produced and the mouth articulates across the spoken line). No handler change since
  v0.1.5; cut to the stable v1.0.0 line as part of the constellation-wide milestone. The `v1.0.0` tag
  builds the consumer image.

## v0.1.5

- **fix(handler): frame-gap truncation -- contiguous output numbering + honest lip-sync floor (#26,
  PR #38; root-causes skyphusion-labs/vivijure#702).** The blend loop named each output PNG by its
  source LOOP index and skipped any frame with a degenerate/placeholder bbox (no face detected that
  frame), punching a hole in the `%08d` sequence; `ffmpeg -f image2` stops at the first gap, so ONE
  early no-face frame truncated the whole clip to its opening run (Night_Shift shot_01: 65 frames in,
  3 out, shipped as a 0.17s "4s" clip that vivijure's #697 duration gate then caught; Night_Signal's
  two dialogue shots hit the same defect). Outputs are now numbered by a contiguous counter (a dropped
  frame can never punch a hole), and a new honest floor (`LIPSYNC_MIN_FRAME_RATIO`, default 0.5)
  degrades to the ORIGINAL full-length clip (`ok:false` + `detail`, no artifact, no error) when the
  face is detectable in fewer than half the frames -- a mostly-faceless shot ships un-synced at full
  length instead of as a stutter. GPU-verified on the exact production inputs: the truncation victim
  (6/64 face frames) degrades honestly; a clean speaking shot is byte-identical to the known-good
  sync. No dependency or base-image change (handler-only release).

## v0.1.4

- **fix(handler): stop the audio-mux from re-encoding the lip-synced video (vivijure #584).** The
  encode path writes a CRF-18 `temp.mp4`, then muxed the audio back in with a second `ffmpeg` call
  that specified no video codec. ffmpeg re-encodes by default, so that mux silently re-ran libx264 at
  its default (~CRF 23, roughly 2 Mbps at 48fps 720p), discarding the CRF-18 first pass and starving
  the mouth region MuseTalk had just generated; an anime 2x upscale downstream then magnified the
  seams (the "breathy" look). The mux now stream-copies the video (`-c:v copy`) and encodes only the
  audio, so the CRF-18 quality reaches the output intact. No double-encode, no bitrate starvation.
