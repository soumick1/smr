# smr_updates165 — ablations: what, how long, one command each

## Why it is cheaper than it looks
Every knob a reviewer will ask about lives in the memory's POLICY, not in the backbone, and the backbone passes are
cached: a policy variant over the seven 7-Scenes sequences costs ~1 min, over 12 CO3D orbits ~3 min. Only the
window-size variants need new passes (~1 min per sequence). Total for the full memory table below: ~2.5 h on one GPU
(w16/w64 included), 45 min without them. The NVS read table: ~70 min.

## Table A — memory policy (scripts/ablate_all.sh -> paper/tab_ablation.tex, 15 variants x {7-Scenes, CO3D})
default | sites 1 / 3 | closure applied as jump / distributed / not applied | batch solve without outlier rejection |
closure sites re-measured | proposals from the RGB cue instead of DINOv2 | scaffold N_h 512 / 8192 | torus 16 / 64 |
windows W=16/8 and 64/32.  Columns: 7-Scenes ATE raw / +SMR / +SMR+PGO and accepted closures per sequence (VGGT-Omega,
Table 2 configuration), CO3D AUC@30 raw / +SMR / +SMR+PGO at N=200 (VGGT, first 12 orbits; ALL=1 for 37).
```bash
cd ~/smr && unzip -o smr_updates165.zip && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/ablate_all.sh                                      # resumable; skips finished sequences
python scripts/ablate_table.py                                                          # prints the summary, writes paper/tab_ablation.tex
```
Run subsets while thinking: `VARIANTS="default sites1 corr_none no_robust" DATASETS=7scenes bash scripts/ablate_all.sh` (~4 min).

## Table B — the in-window read (scripts/nvs/ablate_read.sh, VGGT head, first 300 GSO objects, same objects everywhere)
reads 1 / 2 / 4 / 8 | reads 4 without the symmetric re-measure (--no-align) | reads 4 abstaining instead of the
median (--fallback none).  ~70 min on one GPU; `N=1033` for the full set (~4 h).
```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/nvs/ablate_read.sh      # -> outputs/nvs/ablate/summary_vggt.txt
```

## Predictions (recorded now)
P59: sites=1 accepts more closures but ATE worse than sites=2 on >=4 of 7 scenes (two-site consensus is the gate).
P60: corr_none == raw within noise (proposals alone change nothing: the correction is the read, not the recognition).
P61: no_robust worse than default on at least one scene by >20 % (an outlier edge poisons the batch).
P62: desc_rgb accepts <= half the closures of DINO (the code comment: 1 vs 10 on chess).
P63: N_h 512 -> fewer accepted closures (capacity); N_h 8192 == default within noise. Torus 16 aliases (worse), 64 == default.
P64: W=16 raw worse than W=32 raw; +SMR recovers a larger fraction at W=16 (more junctions, more to repair).
P65: reads 2 < 4 < 8 monotone for VGGT with diminishing returns (8 within +0.1 dB of 4); no-align loses >= half the
     read gain; abstain loses Gaussians and PSNR (v148 finding).
Send outputs/ablate + outputs/nvs/ablate/summary_vggt.txt (or the two tables) and I write the appendix section.
