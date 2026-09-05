# smr_updates139 — NVS (VGGT Table 7 protocol): data pipeline

## 1. Protocol (from the VGGT and LVSM papers)
- LVSM object protocol: **4 input views + 10 target views per object**, GSO (1,099 objects), PSNR/SSIM/LPIPS, 256²
  (VGGT-NVS 224²). Targets given as Plücker rays. Training: LVSM rendered 32 views of 730K Objaverse objects (Adobe,
  never released); VGGT used *"a similar internal dataset of approximately 20 % the size of Objaverse"* (also internal).
  Published rows: LGM 21.44/0.832/0.122, GS-LRM 29.59/0.944/0.051, LVSM 31.71/0.957/0.027, VGGT-NVS 30.41/0.949/0.033.
- Consequence: none of the renders behind Table 7 are public; our table is our own protocol with the published rows as
  caption anchors. Our training set: **~20K Objaverse-LVIS objects (~3 % of Objaverse), stratified over 1,156 categories**,
  rendered by us; GSO rendered by us with the same renderer (train/test rendering consistency). The fraction is stated
  in the caption exactly as VGGT stated its 20 %.

## 2. Our protocol (implemented in `src/smr/nvs/cameras.py`, tested)
Object: bbox centre at the origin, scaled so the bounding sphere has radius 0.5. Cameras: sphere of radius 2.0, vertical
FOV 40° (the object always fits: the ball subtends 29°), 256², OpenCV c2w in a y-up world. Training views: azimuth
U(0,360), elevation U(−20,60). GSO: inputs at elevation 20°, azimuths 0/90/180/270; 10 targets drawn as training views
with a per-object seed. Renderer: Blender (bpy) Cycles GPU, 32 adaptive samples + denoise, white world + one sun (fixed
per object), "Standard" view transform, PNG. Plücker rays provided for later target conditioning.

## 3. Code (v139)
- `src/smr/nvs/cameras.py` + `tests/test_nvs_cameras.py`: look-at, intrinsics, sampling, Plücker, JSON; projection checks.
- `scripts/nvs/render_bpy.py`: headless renderer, resume-safe, batched, per-object seed (crc32 of the id), FAILED markers,
  s/view logged. **VERIFY-ON-SERVER** (bpy 4.x API). `tests/test_render_bpy_mock.py` verifies the geometry with a mock
  bpy: camera looks at the origin and is upright in Blender's z-up space; normalisation lands every vertex within 0.5.
- `scripts/nvs/objaverse_fetch.py` (objaverse package, LVIS stratified, resume), `scripts/nvs/gso_fetch.py` (Fuel API;
  `.zip` download form per Fuel docs), `scripts/nvs/render_all.sh` (3 GPU shards + GSO).

## 4. Dry run first (GPU 0, ~15 min) — then look at ONE image
```bash
cd ~/smr && unzip -o smr_updates139.zip && source .venv/bin/activate
pip install objaverse                      # HF-backed downloader
python tests/test_nvs_cameras.py && python tests/test_render_bpy_mock.py
python scripts/nvs/objaverse_fetch.py --n 20000 --dry-run 20 --index data/nvs/objaverse_index_dry.json
python scripts/nvs/gso_fetch.py --limit 5 --index data/nvs/gso_index_dry.json
CUDA_VISIBLE_DEVICES=0 python scripts/nvs/render_bpy.py --list data/nvs/objaverse_index_dry.json --out ~/data/nvs/dry_objaverse --protocol train --n-views 8 --device OPTIX
CUDA_VISIBLE_DEVICES=0 python scripts/nvs/render_bpy.py --list data/nvs/gso_index_dry.json --out ~/data/nvs/dry_gso --protocol gso --obj-up Z --device OPTIX
grep "s/view" ~/data/nvs/dry_*/render_shard0.log | tail -25
```
If OPTIX fails: `--device CUDA`. Then open `~/data/nvs/dry_gso/<first>/000.png` (an input view: object seen from 20° above,
white background, filling ~70 % of the frame) and `~/data/nvs/dry_objaverse/<first>/000.png`. If a GSO object appears
lying on its side, rerun that scene with `--obj-up Y` and tell me — the up-axis choice is the one thing I cannot verify here.

**Predictions:** P44 render speed 0.25–0.6 s/view at 256²/32 spp on an RTX 6000 Ada (import + normalise 1–3 s/object);
P45 objaverse download 20 GLBs in < 2 min (median 2–8 MB), GSO 5 models in < 1 min; ≤ 1 of 25 assets FAILED.

## 5. Full data run (after the dry run passes)
```bash
python scripts/nvs/objaverse_fetch.py --n 20000                # ~60–160 GB GLBs, hours (HF); keep an eye on disk
python scripts/nvs/gso_fetch.py                                # ~1,030 models, ~15 GB, ~1.5 h at 3 MB/s
bash scripts/nvs/render_all.sh                                  # NVIEWS=16 default: 320K images (~30 GB); GSO 14 views
```
16 views/object over 20K objects (rather than LVSM's 32) trades views for object diversity; the head trains on
(4 input, 4 target) draws so 16 suffices. Predicted wall time at 0.4 s/view: ~13 h on 3 GPUs (P46); GLBs can be deleted
after rendering if disk gets tight (`rm -r ~/data/nvs/objaverse_glb` — the index keeps ids).

## 6. Next (v140): the Gaussian head
Per-pixel 3D Gaussians from frozen-backbone features + point maps (offset, scale, rotation, opacity, colour), gsplat
rendering, MSE + LPIPS, one head per backbone (VGGT, π³, one DUSt3R-family); evaluation on GSO with raw geometry and with
the in-window read (K orderings, symmetric re-measure, consensus on Gaussian positions). Rows: backbone / +SMR read /
+SMR+PGO (= read, one window). Prediction P43 (recorded): +0.3–1.0 dB PSNR for VGGT; identity for π³/MASt3R; we do not
approach LVSM's 31.7 (frozen backbone, light head, 3 % of the data).
