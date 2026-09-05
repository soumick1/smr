# smr_updates125 — the in-window read works; two defects fixed; producing the table rows

## 1. Scored (scoreboard 9.0/20)

| block | result | verdict |
|---|---|---|
| F `ph_reads4` (4 reads, consensus) | **0.528 / 0.387 / 0.458** vs single 0.681 / 0.440 / 0.560 (−18 % Overall) | **P17 hit** |
| F `ph_reads4_ca` (reads → read 0's scale) | 0.654 / 0.293 / 0.473 — Acc *worse* than plain reads | defect 1 |
| F `dp_reads4_ca` (depth branch) | 1.010 / 0.286 / 0.648 vs single 0.968 / 0.295 / 0.631 | depth residual is ordering-independent |
| G chained / smr / smr_pgo + content | **0.585 / 0.608 / 0.604** vs 0.867 / 0.828 / 0.796 | **P18 hit**: 92 % of the window loss recovered; factors 0.987–0.993 per window |
| H relief | 1.064 → **0.261** (reads disagreed by 15–20 % in scale) | P19 hit |
| H terrains | 0.933 → 0.704 (fused kept 27 % of pixels) | near |
| H meadow | 2.341 → 2.387 (all four reads agree with each other) | deterministic failure |
| H courtyard | 0.163 → 0.198 (Comp 0.125 → 0.249; fused kept 47 % of pixels) | defect 2 |
| I scan29 point head | GT cams 1.903, C>1.5 1.864, **no filter 0.881** (C>2: 1.953) | **P20 answered**: threshold holes; the confidence is anti-correlated with accuracy on hard scans |

Findings: the **point head + repeated reads** is the memory-friendly branch (the depth head's residual does not average out); the **content re-measure** is the right correction for windows; the **confidence filter must go** for the point head on our scan set.

## 2. Two defects the logs exposed, fixed in v125

1. **Reference bias in `--content-align`.** All reads were scaled onto read 0. Read 0 is an arbitrary read with its own scale bias; the plain median of four differently-scaled reads averaged the biases and was nearer the truth (Acc 0.528 vs 0.654). Fix: after the pairwise factors are known, renormalise so the geometric mean of the applied depth scales is 1 — spread removed, consensus scale preserved, no pass privileged (`--content-ref mean`, default; `first` = old behaviour). Verified on the fake: an exact read and a +3 % read both land on √1.03.
2. **Consensus drop-outs.** Where fewer than `m` reads agreed, the pixel was dropped (courtyard: 854k of 1.8M pixels; terrains 497k), so a read could be *emptier* than a single pass. Fix: fall back to the earliest witness's point (default; `--no-fallback` = old behaviour). Guarantee: the read is never emptier than the reference pass; where witnesses agree it is more accurate.

Also fixed: a variable-shadowing bug that silently disabled the renormalisation (caught by the dry run's toy check).

## 3. Runs — these produce the paper rows

```bash
cd ~/smr && unzip -o smr_updates125.zip && source .venv/bin/activate && python tests/test_points_suite_dryrun.py | tail -1
runF() { gpu=$1; scan=$2; tag=$3; log=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu python experiments/points_suite.py --gt data/gt/dtu/scan${scan}.npz --backbone vggt \
    --w 49 --overlap 0 --k-ctx 1 --stride 1 "$@" --out-dir outputs/points/dtu_s${scan}_v125_$tag 2>&1 | grep -E "symmetric|fell back|fused:" | tee -a $log
  echo "== $tag scan$scan ==" | tee -a $log
  python scripts/dtu_eval.py --pred outputs/points/dtu_s${scan}_v125_$tag/fused.ply --scan $scan \
    --sampleset ~/data/dtu/SampleSet/MVS\ Data --points-dir ~/data/dtu/Points/stl --icp 8 | tee -a $log; }
```

### J. scan1 recipe check (GPU 1, ~15 min) — P21
```bash
L=outputs/reports/dtu_v125.log
runF 1 1 ph_nofilter            $L --points-from pointhead
runF 1 1 ph_nofilter_reads4     $L --points-from pointhead --reads 4
runF 1 1 ph_nofilter_reads4_ca  $L --points-from pointhead --reads 4 --content-align
runF 1 1 ph_c2_reads4_ca_mean   $L --points-from pointhead --conf-abs 2.0 --reads 4 --content-align
```
**P21:** `ph_c2_reads4_ca_mean` Acc ≤ 0.52, Comp ≤ 0.32, Overall ≤ **0.43** (beats both 0.458 and 0.473). `ph_nofilter` single within ±0.05 of 0.560 on this easy scan; `ph_nofilter_reads4_ca` within ±0.03 of the C>2 variant. Recipe for K = whichever of {nofilter, c2} wins Overall.

### K. Table 2 rows, 23 scans (GPU 1 after J, ~2.5 h) — P22
```bash
PH="--points-from pointhead"                       # switch to "--points-from pointhead --conf-abs 2.0" if J says so
L=outputs/reports/dtu_table2.log
for f in data/gt/dtu/scan*.npz; do s=$(basename $f .npz | tr -d scan)
  runF 1 $s vggtp_single $L $PH
  runF 1 $s vggtp_reads4 $L $PH --reads 4 --content-align
done
python scripts/dtu_summary.py $L
```
**P22:** 23-scan mean Overall: `vggtp_single` ≤ 0.85 (was 1.068 with C>2); `vggtp_reads4` ≤ **0.70**; reads better than single on ≥ 20/23 scans. Rows: "VGGT-p (our harness)" and "+SMR read (4)".

### L. Table 3 rows, 13 scenes, 10-frame protocol (GPU 0, ~40 min) — P23
```bash
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt $f --backbone vggt --n-views 10 --view-seed 0 \
    --w 10 --overlap 0 --k-ctx 1 --stride 1 --reads 4 --content-align --save-maps \
    --out-dir outputs/points/eth_t3_${sc} 2>&1 | grep -E "symmetric|fell back|fused:" | tee -a outputs/reports/eth_table3.log
  echo "== $sc ==" | tee -a outputs/reports/eth_table3.log
  for src in single fused; do
    python scripts/eth3d_pointmap_eval.py --maps outputs/points/eth_t3_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc \
      --gt official --source $src --json outputs/reports/eth_table3.jsonl | grep "^umeyama " | sed "s/^/$src /" | tee -a outputs/reports/eth_table3.log
  done
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_table3.jsonl
```
(`single` = read 0 = the protocol row 0.419 / 0.601 / 0.510; `fused` = "+SMR read".)
**P23:** fused plain-Umeyama 13-scene mean Overall ≤ **0.42** (from 0.510); fused Comp never worse than single by more than 0.02 on any scene (fallback guarantee); relief ≤ 0.30, courtyard ≤ 0.17.

### M. Table 3 beyond-window study, all views in 16-view windows (GPU 2, ~1 h) — P24
```bash
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  for v in raw ca; do [ $v = ca ] && CA="--content-align" || CA=""
    CUDA_VISIBLE_DEVICES=2 python experiments/points_suite.py --gt $f --backbone vggt --w 16 --overlap 8 --k-ctx 0 \
      --stride 1 $CA --save-maps --out-dir outputs/points/eth_win_${sc}_$v 2>&1 | grep -E "symmetric|fused:" | tee -a outputs/reports/eth_windows.log
    echo "== $sc $v ==" | tee -a outputs/reports/eth_windows.log
    python scripts/eth3d_pointmap_eval.py --maps outputs/points/eth_win_${sc}_$v/maps.npz --scene-dir ~/data/eth3d/$sc \
      --gt official --source fused --json outputs/reports/eth_windows_$v.jsonl | grep "^umeyama " | tee -a outputs/reports/eth_windows.log
  done
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_windows_raw.jsonl; python scripts/eth3d_pm_summary.py outputs/reports/eth_windows_ca.jsonl
```
**P24:** content re-measure improves the windowed Overall on ≥ 9/13 scenes and the mean by ≥ 15 %.

## 4. What the tables will say (if P22–P24 hold)
- Table 2: published VGGT 0.382 (their protocol, unreproducible) | KIT feed-forward 0.59 / 0.97 | ours VGGT-p ~0.8, VGGT-d 0.741 | **+SMR read** ~0.7 — the only feed-forward, GT-free improvement row.
- Table 3: published 0.677 | ours protocol row 0.510 (unmasked 0.619) | **+SMR read** ~0.42; beyond-window: windows raw vs + re-measure.
- The story: inside a window the memory is a repeated, re-measured, consensus read; beyond it, the re-measure fixes what camera placement cannot see (per-pass scale). Pose-level SMR/PGO stays inert here by the pass-through invariant — stated as the floor.
