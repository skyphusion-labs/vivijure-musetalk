# Releases -- vivijure-musetalk

The MuseTalk lip-sync image ships as a **2-image split** so a routine (handler-only) release is fast.

## The two images (SAME GHCR package)

Both are tags in `ghcr.io/skyphusion-labs/vivijure-musetalk` -- same package on purpose: GHCR
FROM-layer dedup on push is same-package only, so the consumer push stays handler-layer-only (a
separate `-base` package would re-upload every layer on each release; S19 hard-won fact).

| tag | what | built by | rebuilt when |
|---|---|---|---|
| `base-<N>` | the stable, expensive half: cu128 base + source-compiled openmmlab stack (mmcv/mmpose/mmdet) + MuseTalk checkout + ~7.3GB baked weights | `.github/workflows/base-build.yml` (**dispatch-only**) | ONLY on a dep/weight change (bump `-<N>`) |
| `<X.Y.Z>` / `latest` / `<sha>` | the consumer: `FROM base-<N>@sha256:<digest>` + `COPY handler.py` | `.github/workflows/build-image.yml` (tag `v*` / push to main / dispatch) | every release |

## Cut a release

| What changed | Steps |
|---|---|
| **handler only** (the common case) | Push tag `v<X.Y.Z>`. `build-image.yml` builds the consumer (base layers dedup) and pushes `:<X.Y.Z>` + `:latest` + `:<sha>`. Fast -- only the handler layer uploads. |
| **deps / weights / base image** | 1. Bump `base-<N>` and dispatch `base-build.yml` (stages ~7.3GB, ~35 min on ubuntu-latest). 2. Repin the consumer `Dockerfile` `FROM ...:base-<N>@sha256:<digest>` to the digest the run prints (its step summary). 3. Push the release tag. |

The base is **digest-pinned** in `Dockerfile`, so a consumer build is deterministic and a stale or
renamed base can never be silently substituted. The base build is dispatch-only and deliberate -- never
fork-reachable, never on a routine commit.

## Why a split (and why only here)

Among the vivijure finish satellites only musetalk earns a split: its ~35 min build re-downloads 7.3GB
of weights and recompiles the openmmlab stack cold on every release. upscale / audio-upscale are
base-image-dominated (weights are tens/hundreds of MB) and already dedup same-package; the local-gpu
doors bake no weights at all (runtime pull). The backend seed/runtime/consumer chain (bin-packed
sub-10GB layers, R2 staging, big-runner snapshots, monthly re-bake cadence) is overkill at this scale --
musetalk needs neither R2 nor a big runner and its weights fit one sub-10GB layer. Full evaluation:
vivijure#537.
