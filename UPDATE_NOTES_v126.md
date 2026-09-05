# smr_updates126 — evaluator hardened; the K run is invalid; re-score everything on CPU

## 1. Scored (scoreboard 10.0/24)

- **P21 half.** `ph_c2_reads4_ca_mean` 0.521 / 0.369 / **0.445** — beats reads4 (0.458) and reads4-first (0.473) as predicted; Acc 0.521 (≤0.52 ✓), Comp 0.369 (≤0.32 ✗), Overall 0.445 (≤0.43 ✗ by 0.015).
- **P22 miss — the K run is invalid.** My notes defaulted K to the no-filter point head on the strength of one scan (29). Without the C>2 cut the cloud is ~40 % table/background, and `dtu_eval.py`'s ICP — 60k random points from the whole cloud, 80th-percentile trim — was captured by it on most scans: |ICP scale−1| 5.8 %, |t| 44 mm on average, scan110 scale 0.79 / 153 mm, scan1 0.898 / 76 mm. `dtu_table2.log` measures alignment failure, not point quality; both rows in it are void.
- **P23 miss.** Table-3 reads with the v125 fallback: 0.490 → 0.484. relief, which block H took 1.064 → 0.261, stayed at 0.996: the "earliest witness" fallback re-inserted read 0's failing pixels (fused n_corr identical to single). Read 0 is an arbitrary read; on relief it is the bad one. The earliest-witness rule was the reference bias I removed from content-align, re-introduced through the back door.
- **P24 half.** Beyond-window ETH3D (all views, 16-view windows): raw 0.467 → **0.424** (−9 %), better on 8/13 (predicted ≥9/13, ≥15 %). delivery_area 1.307 → 0.846, office 0.042 → 0.024, relief 0.128 → 0.104; terrace/playground worse. Scenes ≤16 views are one window and identical, as they must be. **relief with all 31 views is 0.104 vs 0.97 under the 10-frame protocol** — VGGT's relief failure is a small-N failure.

## 2. Fixes in v126 (all tested)

1. **`scripts/dtu_eval.py` — the ICP is hardened.** (i) Only prediction points inside the official evaluation region (BB ∧ ObsMask) participate — background is never scored, so it must not steer the gauge; (ii) correspondences beyond MaxDist are excluded each iteration, then the 80 % trim; (iii) **point-to-plane** Sim(3) with GT normals (local PCA on the STL subsample) replaces point-to-point, with early stopping — point-to-point needed >40 iterations to recover a 1 % scale error, point-to-plane recovers it to the noise floor; (iv) a "refinement" that changes scale by >5 % or moves the cloud by >30 mm is **REJECTED** and the camera alignment kept (printed). `tests/test_dtu_eval_icp.py`: 40 % background + 6 mm / 1 % misalignment → Acc 3.69 → 0.29 (noise floor 0.3); an 80 mm shift → REJECTED. **Consequence: every DTU number so far was scored with a weaker gauge; all clouds must be re-scored (CPU only, block N) before any comparison.** `--icp N` is now a maximum (use 50); `--icp-mode point` reproduces the old estimator.
2. **`--fallback median|first|none`** (default **median**): where fewer than m witnesses agree, return the median of all witnesses — symmetric, robust, never emptier than a pass. `first` = v125, `none` = abstain (v124; most accurate, loses completeness).
3. `scripts/dtu_summary.py` parses the new ICP lines and counts rejections. `scripts/dtu_rescore_all.sh` re-scores every DTU cloud on disk in four parallel CPU groups.

## 3. Runs

### N. Re-score every DTU cloud with the hardened gauge (CPU only, 4 parallel groups, ~40 min) — P26
```bash
cd ~/smr && unzip -o smr_updates126.zip && source .venv/bin/activate
python tests/test_dtu_eval_icp.py | tail -1 && python tests/test_points_suite_dryrun.py | tail -1
bash scripts/dtu_rescore_all.sh                # -> outputs/reports/rescore/*.log + combined summary
```
Groups: v123 baselines (ph C>2, dp GT-cams+geo3), K's no-filter single and reads4, and the scan1 variants (native, reads, content, matrix rows) plus scan29.
**P26:** with the hardened gauge (a) `ph_nofilter_single` 23-scan mean Overall ≤ 0.80 and ≤ 3 rejections; (b) `ph_nofilter_reads4` ≤ 0.70 and better than single on ≥ 18/23; (c) `ph_c2_single` (was 1.068) ≤ 0.95 — the holes remain, the gauge improves; (d) `dp_geo3_single` (was 0.741) ≤ 0.70; (e) scan1 `ph_c2_native` 0.560 → ≤ 0.50 and `ph_c2_reads4_ca_mean` 0.445 → ≤ 0.40. Whichever of (a)/(c) wins becomes the Table-2 single row, its reads4 the "+SMR read" row; if (a) wins and (b) holds, Table 2 is done without further GPU.

### O. ETH3D fallback decision, 3 scenes (GPU 0, ~8 min) — P25
```bash
for sc in relief terrains courtyard; do for fb in median none first; do
  CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt data/gt/eth3d/${sc}.npz --backbone vggt --n-views 10 --view-seed 0 \
    --w 10 --overlap 0 --k-ctx 1 --stride 1 --reads 4 --content-align --fallback $fb --save-maps \
    --out-dir outputs/points/eth_fb_${sc}_$fb 2>&1 | grep -E "fallback" | tee -a outputs/reports/eth_fallback.log
  echo "== $sc $fb ==" | tee -a outputs/reports/eth_fallback.log
  python scripts/eth3d_pointmap_eval.py --maps outputs/points/eth_fb_${sc}_$fb/maps.npz --scene-dir ~/data/eth3d/$sc \
    --gt official --source fused --json outputs/reports/eth_fallback.jsonl | grep " Acc " | tee -a outputs/reports/eth_fallback.log
done; done
```
**P25:** relief: none ≈ 0.26, median ≤ 0.45, first ≈ 1.0; courtyard: median ≤ none (completeness); terrains: median ≤ 0.6. Default stays `median` unless `none` wins Overall on all three.

### P. Table-3 "+SMR read" rows, 13 scenes, with the chosen fallback (GPU 0 after O, ~25 min) — P23b
```bash
FB=median   # from O
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt $f --backbone vggt --n-views 10 --view-seed 0 \
    --w 10 --overlap 0 --k-ctx 1 --stride 1 --reads 4 --content-align --fallback $FB --save-maps \
    --out-dir outputs/points/eth_t3b_${sc} 2>&1 | grep -E "fallback" | tee -a outputs/reports/eth_table3b.log
  echo "== $sc ==" | tee -a outputs/reports/eth_table3b.log
  for src in single fused; do
    python scripts/eth3d_pointmap_eval.py --maps outputs/points/eth_t3b_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc \
      --gt official --source $src --json outputs/reports/eth_table3b.jsonl | grep "^umeyama " | sed "s/^/$src /" | tee -a outputs/reports/eth_table3b.log
  done
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_table3b.jsonl
```
**P23b:** fused mean Overall ≤ 0.44 (from 0.490); relief ≤ 0.45; no scene's fused Overall worse than single by > 0.02.

## 4. Where the tables stand
- Table 2: awaiting N. Anchors unchanged (published 0.382 unreproducible; KIT 0.59 / 0.97).
- Table 3: protocol row 0.490 / 0.510 (two runs; nondeterminism ~0.02); beyond-window rows: raw windows 0.467 → re-measure 0.424; "+SMR read" row awaiting P.
- Story unchanged: in-window read = repeated, re-measured, consensus; beyond-window = re-measure of per-pass scale; pose-level SMR/PGO inert by the pass-through invariant (floor).
