# Third-party models (Hub / baked image)

The `ghcr.io/skyphusion-labs/vivijure-musetalk` image bakes MuseTalk and its inference weights so a
worker runs offline. This is the Hub-facing summary. The full copyright and license text for every
component is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

| Role | Component | License | Source |
| --- | --- | --- | --- |
| Lip-sync engine | MuseTalk (TMElyralab) | MIT | https://github.com/TMElyralab/MuseTalk |
| VAE | stabilityai/sd-vae-ft-mse | MIT | https://huggingface.co/stabilityai/sd-vae-ft-mse |
| Audio features | OpenAI whisper-tiny | MIT | https://github.com/openai/whisper |
| Pose | DWPose (dw-ll_ucoco_384) | Apache-2.0 | https://github.com/IDEA-Research/DWPose |
| Face parse | BiSeNet (79999_iter.pth) | MIT | https://github.com/zllrunning/face-parsing.PyTorch |
| Backbone | torchvision resnet18 | BSD-3-Clause | https://github.com/pytorch/vision |
| Pose stack | OpenMMLab mmcv / mmpose / mmdet | Apache-2.0 | https://github.com/open-mmlab |

Wrapper code in this repository is **AGPL-3.0** (see `LICENSE`). None of the baked models carries a
non-commercial restriction.
