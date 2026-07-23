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
