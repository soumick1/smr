# smr_updates142 — NVS core: RGBA renders, GSO listing union, geometry module, Gaussian head, trainer, GSO evaluator

## 0. Apply BEFORE the full render (two data-side changes)
- `render_bpy.py` now writes **RGBA with a transparent film**: alpha = object mask (loaders composite on white; the
  colours are unchanged). The Gaussian head must not turn background pixels into Gaussians, and the mask is the clean
  way to say which pixels are object. If `render_all.sh` was already started from v141: stop the screens
  (`screen -X -S ren0 quit` etc.), `rm -rf ~/data/nvs/objaverse ~/data/nvs/gso`, restart with v142 (minutes lost, not hours).
  The loader still accepts RGB-only renders (mask = "not white") so the dry runs stay usable for smoke tests.
- `gso_fetch.py` lists through **all** routes and takes the union (collection search returned 299 of 1,030): the
  "listing via/union" lines tell us how many GSO objects we evaluate on.

## 1. Geometry (`src/smr/nvs/geometry.py`, tested: `tests/test_nvs_geometry.py`)
- Known-input-camera placement (GS-LRM/LVSM "Known Input Cam" setting): each input view's predicted depth (or point
  head, moved to that view's camera) is unprojected through the GT camera; metric scale from the Sim(3) fitting the
  four predicted cameras to the four GT cameras. Maps resampled 518^2 -> 256^2 (square, uncropped: grids align).
- The read: K orderings -> content_align (scale about the GT cameras, symmetric reference) -> consensus (median).
  Same functions as DTU/ETH3D (imported from points_suite). Stub test: scale recovered exactly (1/3), surface error
  26.7 mm -> 16.9 mm with 4 orderings at equal coverage, 1 + 4 backbone calls.
- `head_input`: 8 channels (rgb, xyz, depth, conf) + valid mask (object alpha AND finite geometry).

## 2. Head + renderer (`src/smr/nvs/gaussian_head.py`; torch -- VERIFY-ON-SERVER via the smoke run)
Small U-Net (base 32, ~1.2M params) -> per pixel: offset (bounded 0.02), log-scales (base 0.006 ~ one pixel at the
object), rotation, opacity (starts opaque), colour residual on the input RGB. Zero-initialised output: the head starts
as "splat the backbone's coloured points". gsplat rasterisation (OpenCV cameras, same as ours), white background.
Losses/metrics: MSE + 0.1 LPIPS(VGG); PSNR, SSIM (torchmetrics or built-in), LPIPS.
Install: `pip install gsplat lpips torchmetrics` (gsplat compiles its CUDA kernels at first import, ~2-5 min).

## 3. Trainer / evaluator
- `experiments/train_nvs.py`: one object per step, 4 inputs + 4 targets, raw geometry (reads=1), AdamW 2e-4 with
  warm-up + cosine, grad clip, ckpt/resume, val on a fixed 1 % hold-out, metrics.csv/val.csv/train.log/config.json.
- `experiments/eval_gso.py`: inputs = views 0-3, targets = 4-13; `--reads 1` (raw row) / `--reads 4` (+SMR row);
  jsonl per object (resume-safe), `--summary` prints rows.
- `scripts/nvs/train_all.sh` (vggt / pi3 / mast3r on GPUs 0/1/2), `scripts/nvs/eval_all.sh` (rows; order-invariant
  backbones get reads=1 only: their read is the identity, stated in the caption).

## 4. Smoke test (GPU 0, ~5 min, while the full render runs) — BEFORE training
```bash
cd ~/smr && unzip -o smr_updates142.zip && source .venv/bin/activate
pip install gsplat lpips torchmetrics && python -c "import gsplat, lpips; print('gsplat', gsplat.__version__)"
python tests/test_nvs_geometry.py && python tests/test_render_bpy_mock.py
rm -rf ~/data/nvs/dry_objaverse && CUDA_VISIBLE_DEVICES=0 python scripts/nvs/render_bpy.py --list data/nvs/objaverse_index_dry.json --out ~/data/nvs/dry_objaverse --protocol train --n-views 8 --device OPTIX 2>&1 | grep -v Saved:
CUDA_VISIBLE_DEVICES=0 python experiments/train_nvs.py --backbone vggt --data ~/data/nvs/dry_objaverse --out outputs/nvs/smoke --smoke 2>&1 | tail -30
```
Expect: 20 steps, "gaussians" ~ 50-150K per step (4 views x object pixels), loss falling, ~0.3-0.8 s/step, a val line
(PSNR may be nan if the 1 % hold-out is empty on 20 objects -- fine). Paste the tail; the first torch/gsplat error is
the thing I want to see. Then look at `outputs/nvs/smoke/train.log`.

## 5. Training (after the render finishes; ~3-4 h per head at ~0.4 s/step)
```bash
bash scripts/nvs/train_all.sh              # 30K steps each, GPUs 0/1/2
bash scripts/nvs/eval_all.sh               # rows: <bb> raw / +SMR read on GSO
```
Predictions (recorded): **P43** read: +0.3-1.0 dB PSNR for VGGT over raw; identity for pi3/mast3r. **P47** absolute:
VGGT raw 24-27 dB PSNR on our GSO renders (frozen backbone, light head, ~3 % of Objaverse; LVSM 31.7 with 100 % and a
trained transformer). **P48** pi3 raw >= VGGT raw (better point maps on objects: ETH3D 0.122 vs 0.490).
