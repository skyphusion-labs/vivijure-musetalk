# vivijure-musetalk CONSUMER image (2-image split) -- the thin, fast half.
#
# FROM the digest-pinned base (base.Dockerfile, built + pushed by base-build.yml) + the handler. A
# handler-only release re-pushes ONLY this handler layer; every base layer dedups on GHCR ("layer
# already exists"). The base carries the cu128 stack, the openmmlab build, the MuseTalk checkout, the
# ~7.3GB baked weights, and the runtime ENV (incl. HF_HUB_OFFLINE) -- see base.Dockerfile.
#
# REPIN this digest ONLY on a deliberate base rebuild (dep/weight change -> a new base-<N>): dispatch
# base-build.yml, then paste the @sha256 it prints here. Contract: RELEASES.md.
FROM ghcr.io/skyphusion-labs/vivijure-musetalk:base-1@sha256:9d4e0f25758e85657b6ed1db7a98f65d06e80c50f6533e75ef3e3a7e0b6ec62b

COPY handler.py /app/handler.py
WORKDIR /app/MuseTalk
CMD ["python", "/app/handler.py"]
