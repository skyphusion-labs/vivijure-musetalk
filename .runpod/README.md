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

Third-party model inventory: [THIRD_PARTY_MODELS.md](../THIRD_PARTY_MODELS.md).
