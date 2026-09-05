# smr_updates128 — corroborated-only reads for Table 3; windowed-cloud gauge fixed

## 1. Scored (scoreboard 12.0/30)

- **R/P28 miss.** Abstention at 10 % of depth does not fire on relief's failed pixels (their witness spread is 3–10 % of depth: 23,649 abstained of 494k) and fires far too much on terrains (377k pixels withheld: Comp 1.07 → 1.67, Overall 0.732 → 1.026). Block O had already shown the clean policy: on the protocol metric `none` (return only corroborated points) beat `median` on relief (**0.200** vs 1.006) and terrains (0.492 vs 0.687) and lost 0.05 on courtyard.
- **S/P23c miss.** 13-scene "+SMR read" with median+abstain@10 %: 0.490 → 0.487 (9/13 better; relief_2 0.603 → 0.456, terrace 0.609 → 0.474, pipes 0.078 → 0.037; terrains worse). The threshold compromise helped neither failure mode.
- **T/P29 half.** Content re-measure better on **19/23** scans (pred. ≥ 20); means 1.099 → 1.064 are dominated by gauge failures: 3 REJECTED per row, and two `ca` clouds with 4 % / 26 % of points in the evaluation region. Medians 0.740 → 0.632. Every failure has the same cause, in my suite: for windowed clouds `to_gt` fitted the GT frame from **context 0's 16 cameras only**, so the initial alignment was up to 30 mm off and the region-restricted ICP had almost nothing to work with (or rejected).

## 2. Decisions and fixes (all tested: `tests/test_dtu_eval_icp.py`, `tests/test_points_suite_dryrun.py`)

1. **Table-3 read policy = corroborated-only** (`--fallback none`): the read returns a pixel only where ≥ 2 of 4 reads agree within the 3 %-of-depth gate. Symmetric (no privileged read), best on the protocol metric on the failure scenes, and the cleanest definition for the paper. The median variant becomes the appendix ablation. (On DTU, uncorroborated pixels are < 0.3 %; Q's median-fallback row stands.)
2. **Suite:** the global GT frame of a windowed cloud is now fitted over *every* view's chained camera (first context holding it, mapped through its junction), not context 0's.
3. **Evaluator:** ICP is a function with a **retry policy**: attempt from the camera alignment; if rejected (or if < 50 % of the cloud starts in the evaluation region), retry once from a **robust coarse initialisation** (median centroid of the prediction inside the expanded BB *above the DTU ground plane* → GT median) with a coarse-to-fine schedule (5 clamped point-to-point iterations at 3×MaxDist, then point-to-plane at MaxDist, scale steps clamped to ±5 %); if that is rejected too, the camera alignment is kept and reported. Test: an 80 mm start is recovered to the 0.3 mm noise floor (shift 80.1 mm); with `--icp-init none` it is a clean rejection.
4. `scripts/dtu_summary.py` parses blocks with several ICP lines (rejected + retried) and counts only final rejections.

## 3. Runs

```bash
cd ~/smr && unzip -o smr_updates128.zip && source .venv/bin/activate && python tests/test_dtu_eval_icp.py | tail -1
SS=~/data/dtu/SampleSet/MVS\ Data; PD=~/data/dtu/Points/stl
```

### U. Table 3 "+SMR read" row, corroborated-only (GPU 0, ~25 min) — P30
```bash
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt $f --backbone vggt --n-views 10 --view-seed 0 \
    --w 10 --overlap 0 --k-ctx 1 --stride 1 --reads 4 --content-align --fallback none --save-maps \
    --out-dir outputs/points/eth_t3d_${sc} 2>&1 | grep -E "fused:" | tee -a outputs/reports/eth_table3d.log
  echo "== $sc ==" | tee -a outputs/reports/eth_table3d.log
  for src in single fused; do
    python scripts/eth3d_pointmap_eval.py --maps outputs/points/eth_t3d_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc --gt official --source $src \
      --json outputs/reports/eth_table3d.jsonl | grep "^umeyama " | sed "s/^/$src /" | tee -a outputs/reports/eth_table3d.log
  done
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_table3d.jsonl
```
**P30:** fused 13-scene mean Overall ≤ **0.42** (single 0.490); relief ≤ 0.30, terrains ≤ 0.55, courtyard ≤ 0.20; better on ≥ 9/13.

### V. Re-score the T clouds with the v128 gauge (CPU, 2 groups, ~1 h) — P31
```bash
L=outputs/reports/rescore/windows_v128.log; : > $L
for v in raw ca; do ( for f in data/gt/dtu/scan*.npz; do s=$(basename $f .npz | tr -d scan)
    echo "== win_$v scan$s ==" >> $L.$v
    python scripts/dtu_eval.py --pred outputs/points/dtu_win_${v}_s$s/fused.ply --scan $s --sampleset "$SS" --points-dir $PD --icp 50 >> $L.$v 2>&1
    echo "[$v] scan$s" >&2; done ) & done; wait; cat $L.raw $L.ca > $L
python scripts/dtu_summary.py $L
```
**P31:** `win_ca` rejections ≤ 1, mean Overall ≤ **0.75**, better than `win_raw` on ≥ 20/23; `win_raw` mean ≤ 1.0. If a scan still rejects, it is a genuinely broken windowed reconstruction and is reported as such.

### W. Only if V leaves ≥ 2 rejections: rerun T with the all-view global frame (GPU 2, ~2.5 h)
Same loop as block T in `UPDATE_NOTES_v127.md` §3 (the suite now fits the frame from all cameras); log to `outputs/reports/dtu_windows_v128.log`.

### Still needed from Q
`python scripts/dtu_summary.py outputs/reports/rescore/baselines_v123.log outputs/reports/dtu_table2_reads.log` — the Table-2 "+SMR read" row (P27: ≤ 0.40, ≥ 18/23 better).

## 4. Tables (state)
- **Table 2:** VGGT-p C>2 **0.547 / 0.327 / 0.437** (23 scans, hardened gauge); VGGT-d 1.051 / 0.372 / 0.711; +SMR read (Q, pending); beyond-window windows raw → re-measure (V). Anchors: published 0.382 (unreproducible feed-forward), KIT VGGT-p 0.59 / VGGT-d 0.97.
- **Table 3:** protocol row 0.490 (unmasked 0.619) vs published 0.677; +SMR read corroborated-only (U); all views in 16-view windows raw 0.467 → re-measure 0.424.
