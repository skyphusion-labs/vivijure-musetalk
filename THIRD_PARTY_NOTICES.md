# Third-Party Notices -- vivijure-musetalk

The wrapper code in this repository (the RunPod handler and Dockerfile) is licensed
under **AGPL-3.0** (see `LICENSE`).

The Docker image this repository builds **incorporates and redistributes** the
following third-party software and pretrained model weights, each under its own
license. Their copyright and license notices are reproduced or referenced below as
required. None carries a non-commercial restriction.

| Component | Author / Source | License | Notes |
|---|---|---|---|
| MuseTalk | TMElyralab -- https://github.com/TMElyralab/MuseTalk | MIT | Cloned at build; driven as a subprocess. Weights baked. |
| sd-vae-ft-mse | Stability AI -- https://huggingface.co/stabilityai/sd-vae-ft-mse | MIT | VAE weights baked. |
| Whisper (whisper-tiny) | OpenAI -- https://github.com/openai/whisper | MIT | Audio feature weights baked. |
| DWPose (dw-ll_ucoco_384) | IDEA-Research -- https://github.com/IDEA-Research/DWPose | Apache-2.0 | Pose weights baked. See `licenses/Apache-2.0.txt`. |
| face-parsing (BiSeNet, 79999_iter.pth) | zllrunning -- https://github.com/zllrunning/face-parsing.PyTorch | MIT | Face-parse weights baked. |
| resnet18 backbone | PyTorch / torchvision -- https://github.com/pytorch/vision | BSD-3-Clause | Backbone weights baked. |
| OpenMMLab (mmcv, mmpose, mmdet) | OpenMMLab -- https://github.com/open-mmlab | Apache-2.0 | Installed into the image. See `licenses/Apache-2.0.txt`. |

The authoritative copyright line and full license for each component live at its
source URL above. Full license texts: AGPL-3.0 -> `LICENSE`; Apache-2.0 ->
`licenses/Apache-2.0.txt`; the MIT and BSD-3-Clause templates are reproduced below
(each MIT/BSD component retains its own upstream copyright notice).

---

## MIT License

```
MIT License

Copyright (c) the respective authors of the MIT-licensed components listed above
(TMElyralab / Stability AI / OpenAI / zllrunning), each retaining its own notice.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BSD 3-Clause License

```
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016 (PyTorch/torchvision), all rights reserved,
and the other BSD-3-Clause components listed above, each retaining its notice.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
