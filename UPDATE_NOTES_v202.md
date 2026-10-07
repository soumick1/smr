# smr_updates202 — trained sequence decoders for NVS from the memory map

New: `src/smr/nvs/seq_decoder.py`, `experiments/nvs_seq_cache.py`, `experiments/nvs_seq_decoder.py`, `scripts/nvs_seq_pipeline.sh`.
Unpack at the repo root (paths inside the zip).

What changes for quality (same for both systems):
* native aspect 384 x 288 instead of a 256 x 256 squash, every pixel a Gaussian (no subsampling);
* decoder: 4-level U-Net (about 7M parameters) on 11 channels per pixel (rgb, map xyz, depth, confidence, ray
  direction), offsets and scales relative to the pixel footprint;
* image-space refiner (about 3M) on rendered rgb + alpha + depth, trained jointly;
* L1 + 0.5 LPIPS + 0.2 (1 - SSIM);
* trained in-domain: all 7-Scenes sequences except seq-01 (targets are input frames, their +-2 neighbours removed
  from the sources); tested on the held-out keyframes of the seven seq-01 trajectories;
* one decoder per geometry (raw chain, +SMR), identical architecture / data / schedule / seed; +SMR+PGO and GT
  placement are evaluated with the +SMR decoder.

Run, smoke test first (one training sequence, pumpkin as test, 60 steps, ~10 min; checks every stage):
```bash
cd ~/smr && unzip -o ~/smr_updates202.zip && source .venv/bin/activate
screen -dmS nvsq bash -c 'BB=vggt_omega SMOKE=1 bash scripts/nvs_seq_pipeline.sh 2>&1 | tee outputs/nvsq_smoke.log; exec bash'
```
If `outputs/nvsq_smoke.log` ends with a results table, the full run:
```bash
screen -dmS nvsq2 bash -c 'BB=vggt_omega bash scripts/nvs_seq_pipeline.sh 2>&1 | tee outputs/nvsq_vggt_omega.log; BB=vggt bash scripts/nvs_seq_pipeline.sh 2>&1 | tee outputs/nvsq_vggt.log; exec bash'
```
Caches for both backbones take ~25 min each on the three GPUs, each pair of decoders ~1 h (two GPUs in parallel),
evaluation a few minutes. Results: `outputs/nvs_seq_v2/<bb>/eval.log` (per-sequence PSNR per method and the
table: all targets and revisited targets), renders in `outputs/nvs_seq_v2/<bb>/images/<seq>/`.
Other backbones: `BB=pi3`, `BB=stream3r`, ... (each needs its caches and two decoders). Disk: ~0.4 GB of cache per
sequence and backbone (~10 GB per backbone).
