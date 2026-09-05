# smr_updates127 — Table 2 re-scored; abstention where reads contradict; finishing both tables

## 1. Scored (scoreboard 11.5/27)

**N — re-scored with the hardened gauge (`scripts/dtu_summary.py outputs/reports/rescore/*.log`), 23 scans, 0 rejections:**

| row | Acc | Comp | Overall | old gauge |
|---|---|---|---|---|
| **VGGT-p, C>2, predicted cams** | **0.547** | **0.327** | **0.437** | 1.086 / 1.051 / 1.068 |
| VGGT-p, no filter | 0.775 | 0.304 | 0.540 | (invalid run) |
| VGGT-p, no filter, 4 reads + re-measure | 0.700 | 0.295 | 0.498 | (invalid run) |
| VGGT-d, GT cams, geo≥3, C>2 | 1.051 | 0.372 | 0.711 | 1.093 / 0.389 / 0.741 |

scan1: native 0.560 → **0.291**; reads4 0.269; reads4 + re-measure (symmetric) **0.259**; windows chained 0.711 → + re-measure **0.349** (smr / smr_pgo 0.360 / 0.355). scan29 no-filter 0.543.

**Correction of my block-I diagnosis.** The point head's "coverage defect" (Comp 1.05) was not confidence holes — it was the old point-to-point ICP stalling 5–19 mm off, which inflated Acc *and* Comp; the converged point-to-plane gauge (typical correction 2.4 % scale / 18 mm) removes it entirely. With the proper gauge the no-filter recipe loses (0.540 vs 0.437): **KIT's C>2 stands and is the Table-2 recipe.** P26's numbers hit, P20's mechanism was wrong; the evaluator fix exposed it.

**O — fallback:** relief: none **0.200**, median 1.006, first 0.996; terrains: none 0.492, median 0.687; courtyard: median **0.142**, none 0.193. Abstaining wins when a read fails grossly (dropping the failed pixels also un-drags the Umeyama: relief Comp 1.31 → 0.33); the median wins when the witnesses merely disagree modestly. The spread of the witnesses separates the two cases (metres vs centimetres). P25 half.

**P — Table-3 "+SMR read" (median fallback):** 0.490 → **0.473**, better on 8/13 (pipes 0.078 → 0.037, terrace 0.609 → 0.493, playground 0.294 → 0.235); relief unchanged (1.006) because its failed pixels were kept. P23b miss (≤ 0.44).

## 2. v127 changes
- **Spread-gated abstention** inside the median fallback: where fewer than m witnesses agree *and* the median witness-to-consensus distance exceeds `--abstain-rel` (default 0.10) × depth, the read abstains (NaN) instead of returning a compromise. Reads that contradict each other by metres carry no estimate; the memory declines to answer there. Dry run: a read failing grossly on half of every image → that half withheld (8 % finite), the modestly-disagreeing half kept (100 %); `--abstain-rel 0` never abstains.
- `scripts/dtu_rescore_all.sh`: per-scan progress on stderr, no `set -e` trap (the v126 groups finished but their "done" line never printed).
- **Recipes, final:** DTU = `--points-from pointhead --conf-abs 2.0 --stride 1`, evaluator `--icp 50` (point-to-plane, region-restricted, rejecting); "+SMR read" = `--reads 4 --content-align` (symmetric reference, median fallback with abstention). ETH3D = same read on the protocol's 10 frames; beyond-window = `--w 16 --overlap 8 --k-ctx 0 --content-align`.

## 3. Runs

```bash
cd ~/smr && unzip -o smr_updates127.zip && source .venv/bin/activate && python tests/test_points_suite_dryrun.py | tail -1
SS=~/data/dtu/SampleSet/MVS\ Data; PD=~/data/dtu/Points/stl
```

### Q. Table 2 "+SMR read" row, 23 scans (GPU 1, ~1.5 h) — P27
```bash
L=outputs/reports/dtu_table2_reads.log
for f in data/gt/dtu/scan*.npz; do s=$(basename $f .npz | tr -d scan)
  CUDA_VISIBLE_DEVICES=1 python experiments/points_suite.py --gt $f --backbone vggt --w 49 --overlap 0 --k-ctx 1 --stride 1 \
    --points-from pointhead --conf-abs 2.0 --reads 4 --content-align --out-dir outputs/points/dtu_t2_reads_s$s 2>&1 | grep -E "symmetric|abstained" | tee -a $L
  echo "== ph_c2_reads4 scan$s ==" | tee -a $L
  python scripts/dtu_eval.py --pred outputs/points/dtu_t2_reads_s$s/fused.ply --scan $s --sampleset "$SS" --points-dir $PD --icp 50 | tee -a $L
done
python scripts/dtu_summary.py outputs/reports/rescore/baselines_v123.log $L
```
**P27:** `ph_c2_reads4` 23-scan mean Overall ≤ **0.40** (single 0.437), better on ≥ 18/23 scans, 0 rejections.

### R. Abstention check, 3 scenes (GPU 0, ~8 min) — P28
```bash
for sc in relief terrains courtyard; do
  CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt data/gt/eth3d/${sc}.npz --backbone vggt --n-views 10 --view-seed 0 \
    --w 10 --overlap 0 --k-ctx 1 --stride 1 --reads 4 --content-align --save-maps --out-dir outputs/points/eth_abst_${sc} 2>&1 | grep abstained | tee -a outputs/reports/eth_abstain.log
  echo "== $sc ==" | tee -a outputs/reports/eth_abstain.log
  python scripts/eth3d_pointmap_eval.py --maps outputs/points/eth_abst_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc --gt official --source fused \
    --json outputs/reports/eth_abstain.jsonl | grep " Acc " | tee -a outputs/reports/eth_abstain.log
done
```
**P28:** relief ≤ 0.35 (none was 0.200, median 1.006), courtyard ≤ 0.15 (median was 0.142), terrains ≤ 0.55. If relief stays > 0.5, lower `--abstain-rel` to 0.05 and rerun the three.

### S. Table 3 "+SMR read" row, 13 scenes, with abstention (GPU 0 after R, ~25 min) — P23c
```bash
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt $f --backbone vggt --n-views 10 --view-seed 0 \
    --w 10 --overlap 0 --k-ctx 1 --stride 1 --reads 4 --content-align --save-maps --out-dir outputs/points/eth_t3c_${sc} 2>&1 | grep abstained | tee -a outputs/reports/eth_table3c.log
  echo "== $sc ==" | tee -a outputs/reports/eth_table3c.log
  for src in single fused; do
    python scripts/eth3d_pointmap_eval.py --maps outputs/points/eth_t3c_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc --gt official --source $src \
      --json outputs/reports/eth_table3c.jsonl | grep "^umeyama " | sed "s/^/$src /" | tee -a outputs/reports/eth_table3c.log
  done
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_table3c.jsonl
```
**P23c:** fused mean Overall ≤ **0.43** (single 0.490); relief ≤ 0.40; better on ≥ 9/13.

### T. Table 2 beyond-window study, 23 scans, 16-view windows raw vs + re-measure (GPU 2, ~2.5 h) — P29
```bash
L=outputs/reports/dtu_windows.log
for f in data/gt/dtu/scan*.npz; do s=$(basename $f .npz | tr -d scan)
  for v in raw ca; do [ $v = ca ] && CA="--content-align" || CA=""
    CUDA_VISIBLE_DEVICES=2 python experiments/points_suite.py --gt $f --backbone vggt --w 16 --overlap 8 --k-ctx 0 --stride 1 \
      --points-from pointhead --conf-abs 2.0 $CA --out-dir outputs/points/dtu_win_${v}_s$s 2>&1 | grep -E "symmetric|fused:" | tee -a $L
    echo "== win_$v scan$s ==" | tee -a $L
    python scripts/dtu_eval.py --pred outputs/points/dtu_win_${v}_s$s/fused.ply --scan $s --sampleset "$SS" --points-dir $PD --icp 50 | tee -a $L
  done
done
python scripts/dtu_summary.py $L
```
**P29:** windows raw mean Overall 0.65–0.90; + re-measure ≤ **0.50** (native 0.437); re-measure better on ≥ 20/23.

## 4. Tables after Q/S/T
- **Table 2 (DTU, mm):** published VGGT 0.389/0.374/0.382 (their protocol; not reproduced feed-forward by KIT or us) | KIT feed-forward VGGT-p 0.74/0.44/0.59, VGGT-d 1.53/0.40/0.97 | ours (one harness): VGGT-d 1.051/0.372/0.711; VGGT-p **0.547/0.327/0.437**; **VGGT-p + SMR read** (Q); beyond-window: 16-view windows raw vs + re-measure (T).
- **Table 3 (ETH3D, m):** published 0.873/0.482/0.677 | ours protocol row 0.415/0.565/0.490 (unmasked 0.686/0.553/0.619) | **+ SMR read** (S) | all views in windows: raw 0.530/0.404/0.467 → re-measure 0.502/0.347/0.424.
- Captions carry the gauge (official evaluator; converged point-to-plane Sim(3) refinement, rejection threshold) and the read (4 orderings, symmetric re-measure, consensus with spread-gated abstention). Pose-level SMR/PGO rows stated as inert (floor) with the pass-through invariant.
