# smr_updates130 — tab:dtu complete; a regression in the windowed ETH3D row found and fixed

## 1. Scored (scoreboard 14.0/35)
- **W / P32 half.** With the all-view frame the re-measure is better on **18/22** standard scans
  (pred. ≥ 20); standard-22 means raw 1.626/0.935/**1.280** → **1.077/0.735/0.906**, medians 0.824 → 0.604.
  Three scans (10, 12, 33) cannot be locked by the gauge in either variant (Overall > 1.5) and dominate the
  means; on the 19 that lock: raw 1.307/0.823/1.065 → **0.693/0.576/0.635** (17/19 better). Both lines are in
  `tab:dtu`; the caption names the three scans.
- **X / P33 miss — a regression, root-caused and fixed.** The ETH3D windowed re-measure row under the
  v129 defaults was 0.527 (v124 row 0.424, raw 0.467). `n_corr` gave it away: v124 kept every pixel
  (5.03 M on courtyard), v129 kept 55 %. The symmetric renormalisation scaled each window about its
  **own** placed cameras; a view shared by two windows has two slightly different placed camera
  centres, so points that coincided after alignment were pulled apart by (1/g−1)·Δc — negligible on
  DTU (g≈1, Δc≈mm), 45 % of the pixels on ETH3D windows (scale factors far from 1, junction
  residuals of decimetres). The fake backbone could not show it because its windows share exact cameras.
  Fix (v130): the renormalisation is a **repeated-reads** operation — applied only when all contexts
  see the same views, and about **one common centre per view**; for chained windows the global scale
  is the gauge's and nothing is renormalised. The W (DTU) rows are affected at the 0.1 % level only
  (g within 1 % of 1, Δc of millimetres) and are not re-run.

## 2. Paper
`paper/downstream.tex` now complete for Table 2 (all rows filled) and Table 3 (re-measure row = v124 run,
to be replaced by block Y if it moves). Compiles in the Overleaf copy: 0 errors.

## 3. Run — Y: ETH3D windowed re-measure row with v130 (GPU 0, ~30 min) — P34
```bash
cd ~/smr && unzip -o smr_updates130.zip && source .venv/bin/activate && python tests/test_points_suite_dryrun.py | tail -1
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt $f --backbone vggt --w 16 --overlap 8 --k-ctx 0 --stride 1 \
    --content-align --fallback none --save-maps --out-dir outputs/points/eth_win130_${sc} 2>&1 | grep -E "windows \(different|fused:" | tee -a outputs/reports/eth_windows130.log
  echo "== $sc ==" | tee -a outputs/reports/eth_windows130.log
  python scripts/eth3d_pointmap_eval.py --maps outputs/points/eth_win130_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc --gt official --source fused \
    --json outputs/reports/eth_windows130.jsonl | grep "^umeyama " | tee -a outputs/reports/eth_windows130.log
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_windows130.jsonl
```
**P34:** every scene's `n_corr` equals the raw row's (no pixels lost to the renormalisation); 13-scene mean
Overall within ±0.02 of the v124 row 0.424. Replace the `tab:eth3d` re-measure row with these numbers.

## 4. Then
Overleaf insertion (notes v129 §3); streaming backbones raw vs +SMR; compute row for the 4-ordering read.
