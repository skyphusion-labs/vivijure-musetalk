# RunPod Hub -- Vivijure MuseTalk

Hub listing config for the Vivijure lip-sync finish satellite.

## Required environment (finish-chain / R2 mode)

| Env key | What to put |
| --- | --- |
| `R2_ENDPOINT_URL` | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | Public half of an R2 API token |
| `R2_SECRET_ACCESS_KEY` | Secret half of that token |
| `R2_BUCKET` | Bucket shared with Vivijure Studio (default `vivijure`) |

**Name check:** this worker reads `R2_ENDPOINT_URL`. The main `vivijure-backend` listing uses
`R2_ENDPOINT` (no `_URL`). Copy the four values carefully when wiring both.

## Hub test

`.runpod/tests.json` sends `{ "selftest": true }`. That runs MuseTalk end to end on a baked sample
and does not need R2. Pin **Blackwell** or **Hopper** (CUDA 12.8 image).

## GPU and disk (source of truth: the production endpoint)

`hub.json` mirrors what the Vivijure production endpoint actually runs (`zw6pt4lymf69pk`, running `ghcr.io/skyphusion-labs/vivijure-musetalk:1.0.5`,
read from the RunPod API on 2026-07-25), so a Hub deployer gets the configuration we ourselves
prove every day:

- `gpuIds`: `BLACKWELL_96,HOPPER_141,BLACKWELL_180`. `BLACKWELL_96` (RTX PRO 6000) is the pool production uses; the larger pools stay
  listed as fallbacks for availability. The earlier config excluded `BLACKWELL_96` outright, which
  pushed Hub deployers onto B200 and H200 class cards at roughly two to three times the hourly cost
  for the same job.
- `containerDiskInGb`: `40` (unchanged, and it matches prod). The image is 18.2 GB compressed.
- `tests.json` pins `NVIDIA RTX PRO 6000 Blackwell Server Edition`: the card production runs on, so a
  green Hub test means the same thing our own endpoint means.

Repin this section together with `hub.json` whenever the production endpoint moves pools or image.

Third-party model inventory: [THIRD_PARTY_MODELS.md](../THIRD_PARTY_MODELS.md).
