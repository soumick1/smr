# smr_updates131 — the windowed-fusion policy, decided consistently (no code change)

## 1. Scored (scoreboard 14.0/36) and a corrected diagnosis
- **Y / P34 miss.** ETH3D windowed re-measure row with `--fallback none`: 0.562 (raw-with-first 0.467, v125-first row 0.424).
- **Correction.** The pixel loss (5.03 M → 2.9 M correspondences on courtyard) was **not** a placement regression and
  not the renormalisation (v130's claim): the "v124" beyond-window runs (block M) were executed with v125 code, whose
  default `fallback=first` keeps every pixel (context 0's point where the two windows disagree); X/Y used
  `--fallback none`, which drops every overlap pixel whose two windows disagree beyond 3 % of depth — 45 % on ETH3D
  windows. An A/B of the v124 and current suites on a jittered fake shows no regression (current keeps more).
  v130's scoping of the renormalisation to repeated reads stands (renormalising windows is pointless), but it was
  not the cause.

## 2. Policy, stated once for the paper
- **In-window read** (K orderings of the same views): corroborated-only (`--fallback none`) — a pixel is returned
  where ≥2 of 4 reads agree within 3 % of depth. Best on the protocol metric (Table 3: 0.437) and principled.
- **Windowed fusion** (chained windows, overlap of 8): an overlap pixel has exactly two witnesses; nothing can be
  corroborated against, so the junction consensus is the **median of the windows** (symmetric, all pixels kept):
  `--fallback median --abstain-rel 0`. Both datasets' windowed rows use this; the `none` variants go to the
  appendix as the "corroborated-only" ablation (DTU: 1.280 → 0.906; ETH3D: 0.562).

## 3. Runs (both datasets, same policy)

```bash
cd ~/smr && unzip -o smr_updates131.zip && source .venv/bin/activate
SS=~/data/dtu/SampleSet/MVS\ Data; PD=~/data/dtu/Points/stl
```

### Z1. ETH3D windowed rows, median fusion (GPU 0, ~1 h for raw + re-measure) — P35
```bash
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  for v in raw ca; do [ $v = ca ] && CA="--content-align" || CA=""
    CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt $f --backbone vggt --w 16 --overlap 8 --k-ctx 0 --stride 1 \
      $CA --fallback median --abstain-rel 0 --save-maps --out-dir outputs/points/eth_win131_${v}_${sc} 2>&1 | grep -E "fused:" | tee -a outputs/reports/eth_windows131.log
    echo "== $sc $v ==" | tee -a outputs/reports/eth_windows131.log
    python scripts/eth3d_pointmap_eval.py --maps outputs/points/eth_win131_${v}_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc --gt official --source fused \
      --json outputs/reports/eth_windows131_$v.jsonl | grep "^umeyama " | tee -a outputs/reports/eth_windows131.log
  done
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_windows131_raw.jsonl; python scripts/eth3d_pm_summary.py outputs/reports/eth_windows131_ca.jsonl
```
**P35:** every scene's `n_corr` equals the single-pass count (all pixels kept); raw mean Overall within ±0.03 of
0.467; re-measure ≤ 0.43 (the v125-first row was 0.424) and better than raw on ≥ 8/13.

### Z2. DTU windowed rows, median fusion (GPU 2, ~2.5 h) — P36
```bash
L=outputs/reports/dtu_windows131.log
for f in data/gt/dtu/scan*.npz; do s=$(basename $f .npz | tr -d scan)
  for v in raw ca; do [ $v = ca ] && CA="--content-align" || CA=""
    CUDA_VISIBLE_DEVICES=2 python experiments/points_suite.py --gt $f --backbone vggt --w 16 --overlap 8 --k-ctx 0 --stride 1 \
      --points-from pointhead --conf-abs 2.0 $CA --fallback median --abstain-rel 0 --out-dir outputs/points/dtu_win131_${v}_s$s 2>&1 | grep -E "fused:" | tee -a $L
    echo "== win_$v scan$s ==" | tee -a $L
    python scripts/dtu_eval.py --pred outputs/points/dtu_win131_${v}_s$s/fused.ply --scan $s --sampleset "$SS" --points-dir $PD --icp 50 | tee -a $L
  done
done
python scripts/dtu_summary.py $L
```
**P36:** standard-22 means: raw ≤ 1.20 (was 1.280 with `none`), re-measure ≤ **0.85** (was 0.906), better on ≥ 18/22;
the same three scans (10, 12, 33) may still not lock.

## 4. Then
Update the two beyond-window rows in `paper/downstream.tex` from Z1/Z2 (captions already describe the median
fusion once the word "corroborated" is moved to the read row only); insert into the Overleaf; streaming backbones.
