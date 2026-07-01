# Deploy the lip-sync finish engine

This page walks you through standing up `vivijure-musetalk` on your own. When you finish, you will
have a RunPod endpoint that lip-syncs a character's mouth to spoken dialogue, and an endpoint id you
paste into your Vivijure Studio to turn it on.

New here? The one-page picture of how the parts fit together is in
[constellation.md](constellation.md). This engine is one box on that map.

## What you need first

- A **RunPod** account, and an **API key** from it (runpod.io, then Settings, then API Keys).
  RunPod is where the GPU runs.
- **Docker** on your computer, so you can build the image. (If someone already pushed an image you
  can use, you can skip building; see "Re-running" below.)
- A **registry** to push the image to, and to be logged in to it (for example GitHub Container
  Registry, `ghcr.io`). This is the shelf the endpoint pulls the image from.
- Optional, for the studio's normal mode: **R2 storage keys** (Cloudflare R2). The endpoint reads
  the face clip and audio from R2 and writes the result back to R2.

## The short path

```bash
cp deploy.env.example deploy.env   # then open deploy.env and fill in your keys
./deploy.sh                        # safe to re-run
```

The script builds the image, pushes it, creates the RunPod endpoint, and prints the endpoint id. It
stops on the first error, so you never end up half-deployed.

## What the script does, step by step

1. **Builds** the Docker image from this repo.
2. **Pushes** it to your registry.
3. **Creates a RunPod template and endpoint** (or reuses them if they already exist), pinned to the
   GPU you chose, set to scale to zero.
4. **Prints the endpoint id** and reminds you how to wire it into the studio.

## Every setting you can set

All settings live in `deploy.env`. The example file has them all with comments; here is what each one
means and why.

### The keys you must set

- **`RUNPOD_API_KEY`** -- your RunPod API key. Why: the script talks to RunPod for you to make the
  endpoint. Example: `RUNPOD_API_KEY=rpa_XXXX...`.
- **`IMAGE`** -- the image name to build and run. Why: it is both where the script pushes the image
  and what the endpoint pulls. Point it at your own registry. Example:
  `IMAGE=ghcr.io/yourname/vivijure-musetalk:latest`.
- **`ENDPOINT_NAME`** -- a label for the endpoint. Why: the script finds and reuses an endpoint by
  this name, so re-running is safe. Example: `ENDPOINT_NAME=vivijure-musetalk`.
- **`GPU_TYPE_IDS`** -- which GPU cards RunPod may use, separated by commas. Why: this image is built
  for **CUDA 12.8**, which needs a card on a **new driver**. Pin it to **Blackwell (RTX PRO 6000)** or
  **Hopper (H100 / H200)** cards, which are always on new drivers. A cheap 4090 or L40S host is a
  driver lottery for this image and may refuse to start. Example:
  `GPU_TYPE_IDS=NVIDIA RTX PRO 6000 Blackwell Server Edition,NVIDIA H100 80GB HBM3`.

### The knobs you usually leave alone

- **`CONTAINER_DISK_GB`** (default `30`) -- how much disk the container gets. Why: MuseTalk bakes
  about 7GB of model weights into the image, so it needs room. Example: `CONTAINER_DISK_GB=30`.
- **`WORKERS_MIN`** (default `0`) -- the fewest workers kept running. Why: `0` means scale to zero, so
  you pay nothing when no one is rendering. Example: `WORKERS_MIN=0`.
- **`WORKERS_MAX`** (default `2`) -- the most workers that can run at once. Why: caps how many jobs run
  in parallel (and your spend). Example: `WORKERS_MAX=2`.
- **`IDLE_TIMEOUT`** (default `5`) -- seconds a worker stays warm after a job before it shuts down.
  Why: a small warm window avoids a cold start if a second shot arrives right away. Example:
  `IDLE_TIMEOUT=5`.
- **`EXECUTION_TIMEOUT_MS`** (default `600000`) -- the longest a single job may run, in milliseconds
  (600000 = 10 minutes). Why: a stuck job is cut off instead of billing forever. Example:
  `EXECUTION_TIMEOUT_MS=600000`.
- **`CONTAINER_REGISTRY_AUTH_ID`** (default empty) -- a RunPod credential id for a **private** image.
  Why: if your image is private, RunPod needs a login to pull it. Make one in the RunPod console
  (Settings, then Container Registry Auth) and paste its id here. Leave it blank for a public image.
- **`REGISTRY_USER`** / **`REGISTRY_TOKEN`** (default empty) -- a login for your registry, used to push
  the image. Why: pushing needs you to be logged in. Leave blank if you already ran `docker login`.
- **`SKIP_BUILD`** (default `0`) -- set `1` to skip build and push and reuse an image already pushed.
  **`SKIP_ENDPOINT`** (default `0`) -- set `1` to stop after pushing the image (no endpoint).

### The endpoint's own settings (R2 mode)

The studio's normal mode is "finish-chain" mode: the endpoint reads and writes your R2 bucket by key,
so no clip data passes through the studio. Set these four to turn it on. Leave them blank to use only
the presigned-URL mode, where the studio hands the endpoint short-lived links instead.

- **`R2_ENDPOINT_URL`** -- your R2 S3 address (looks like `https://<account>.r2.cloudflarestorage.com`).
- **`R2_BUCKET`** (default `vivijure`) -- the bucket name the clips live in.
- **`R2_ACCESS_KEY_ID`** / **`R2_SECRET_ACCESS_KEY`** -- an R2 key pair scoped to that bucket. Make a
  key just for this engine so its reach is small.

## What the endpoint expects as a job

You do not call this by hand in normal use; the studio does. But so you know exactly what it does,
here is the contract. It takes **two** inputs (a face clip and an audio track) and gives back a
lip-synced clip.

- **R2 finish-chain mode:** `{ "clip_key": "...", "audio_key": "...", "output_key": "...",
  "bbox_shift": 0, "version": "v15" }`. The endpoint reads both keys from R2 and writes the result to
  `output_key`.
- **Presigned mode:** `{ "video_url": "...", "audio_url": "...", "output_url": "...",
  "output_key": "..." }`. The studio presigns the links; the endpoint holds no keys.
- **Self-test:** `{ "selftest": true }`. Runs MuseTalk end to end on a baked sample face and speech.
  Use it to prove a fresh endpoint works. It doubles as a health check.

Two job knobs you can pass:

- **`bbox_shift`** (default `0`) -- nudges the mouth region box up or down a little. Why: on some faces
  the auto-found box sits slightly off; a small shift lines it up. Leave `0` unless the mouth looks
  misplaced.
- **`version`** (default `v15`) -- which MuseTalk model version to use (`v15` is the newest). Why:
  `v15` gives the best mouth blend; `v1` is the older model. Leave `v15`.

The result is `{ ok, clip_key|output_key, bytes, version, applied: ["lipsync:v15"] }`. If a shot has
no clear face to work on, the engine passes the original clip through unchanged instead of failing, so
a misrouted shot never breaks your film.

## Turn it on in the studio

This engine powers the studio's **finish-lipsync** module (an opt-in tier). To turn it on:

1. Copy the endpoint id the script printed.
2. In your studio's `deploy.env`, set **`MUSETALK_RUNPOD_ENDPOINT_ID`** to that id.
3. Keep `VIVIJURE_PROFILE=full` and re-run the studio's `./deploy.sh`.

Full context on the tiers is in the studio's `docs/opt-in-tiers.md` (the "finish-lipsync" entry). It
works best with `speech-upscale` on, so the lips follow cleaned dialogue.

## Re-running and fixing things

- Re-running `./deploy.sh` is safe. It reuses the template and endpoint it already made.
- To change the endpoint's GPU or scaling after it exists, use the RunPod console; RunPod does not let
  the API re-pin an endpoint's GPU list after creation.
- If a push fails, make sure you ran `docker login` for your registry and that the repo exists there.
- If the endpoint's workers never start, check the GPU pin: a cu128 image on an old-driver host will
  refuse to boot. Pin Blackwell or Hopper.
