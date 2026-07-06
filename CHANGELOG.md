# Changelog -- vivijure-musetalk

The image ships as a git-tag-driven release (`v<X.Y.Z>`; see `RELEASES.md`). Each tag builds the
consumer image. This file records the why behind each release; the tag is the version of record.

## v0.1.4

- **fix(handler): stop the audio-mux from re-encoding the lip-synced video (vivijure #584).** The
  encode path writes a CRF-18 `temp.mp4`, then muxed the audio back in with a second `ffmpeg` call
  that specified no video codec. ffmpeg re-encodes by default, so that mux silently re-ran libx264 at
  its default (~CRF 23, roughly 2 Mbps at 48fps 720p), discarding the CRF-18 first pass and starving
  the mouth region MuseTalk had just generated; an anime 2x upscale downstream then magnified the
  seams (the "breathy" look). The mux now stream-copies the video (`-c:v copy`) and encodes only the
  audio, so the CRF-18 quality reaches the output intact. No double-encode, no bitrate starvation.
