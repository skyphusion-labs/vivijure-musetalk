# Changelog -- vivijure-musetalk

The image ships as a git-tag-driven release (`v<X.Y.Z>`; see `RELEASES.md`). Each tag builds the
consumer image. This file records the why behind each release; the tag is the version of record.

## Unreleased

- Nothing yet.

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
