# Security audit false positives

Documented dismissals for adversarial-audit (K2.7/K3) findings that are not actionable bugs in this repo's threat model.

## Homelab presigned mode

When `R2_URL_HOST_SUFFIX` is unset, presigned GET/PUT targets homelab operator-configured URLs only. Production RunPod templates set the suffix; empty default is homelab convenience, not prod.

## Self-hosted workflow_dispatch

`base-build.yml` workflow_dispatch runs on `[self-hosted, fleet]` with org-member auth only. Input tags are operator-supplied image labels, not public HTTP.

## Record

| Date | Audit | Finding | Rationale |
| --- | --- | --- | --- |
| 2026-07-23 | K3 repo | workflow_dispatch input in shell | Self-hosted dispatch; org-member only |
| 2026-07-23 | K3 repo | code-coverage workflow token scopes | Standard org CI; fork PR guard |
| 2026-07-23 | K3 repo | Presigned SSRF host pin optional | Homelab mode; prod template sets suffix |
| 2026-07-23 | K3 verify ~18:04 | R2 secret as plain string in hub.json | RunPod Hub UI masks secrets at runtime; operator configures template |
| 2026-07-23 | K3 verify ~18:04 | Model weights without integrity pin | Operator bake-time; gdown/HF at Docker build |
| 2026-07-23 | K3 verify ~18:04 | packages:write on non-publish CI | GHCR push gated to tag/release paths |
| 2026-07-24 | K2.7 PR #TBD (#67) | ffmpeg silencedetect on job audio_key | Operator-supplied dialogue WAV from the render chain; local path only; no URL fetch; bounds rest-hold |
