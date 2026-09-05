# smr_updates129 — both tables' main rows are final; two beyond-window rows remain

## 1. Scored (scoreboard 13.5/33)

- **Q / P27 hit.** Table-2 "+\smr{} read" (point head, C>2, 4 orderings, symmetric re-measure, consensus):
  **0.472 / 0.312 / 0.392** over 23 scans, **better than the single pass on 23/23**; standard 22-scan
  split (drop scan6): **0.476 / 0.315 / 0.396** vs single 0.552 / 0.330 / 0.441. Feed-forward, no GT,
  no BA — on par with the published 0.382 that no independent study reproduces (KIT 0.59).
- **U / P30 half.** Table-3 read, corroborated-only: **0.274 / 0.599 / 0.437** vs single 0.415 / 0.565 /
  0.490 (mean pred. ≤ 0.42; relief 0.200 ✓, terrains 0.492 ✓, courtyard 0.193 ✓; 8/13 better, pred. ≥ 9).
  Accuracy halves; Completeness is paid where the reads rarely agree (relief_2: 6 % corroborated →
  1.151; playground 24 % → 0.381). Policy kept: it is the same read as DTU's and the honest one.
- **V / P31 miss.** Windowed DTU re-score: gauge converged on both variants for only 15/23 scans
  (on those raw 0.718 → **0.576**, re-measure better 15/15); 2 + 5 rejections; and my v128
  "coarse-first when < 50 % in region" order made scan23 (0.716 → 3.244) and scan33 worse.

## 2. Fixed in v129
`scripts/dtu_eval.py`: every alignment candidate — camera alignment as-is, plain refinement, coarse-initialised
refinement — is scored by **fit quality** (median NN distance of in-region points to the scan) and the best is
kept; a refinement replaces the camera alignment only if it improves the fit. Order never decides.
Test: far start → coarse candidate chosen at 0.31 mm vs 19.6 mm; with `--icp-init none` a clean rejection.

## 3. Paper snippet (`paper/downstream.tex`)
Drop-in for the uploaded Overleaf: a results subsection `sec:mvs` + `tab:dtu` + `tab:eth3d` in Table-1's
style (booktabs, scriptsize, `\smr{}`). Insert by replacing the line
`\emph{Remaining tasks (multi-view depth, point maps, two-view matching, view synthesis) follow in the results revision.}`
with `\input{downstream}`, and append to `references.bib`:
```
@article{langendoerfer2026vggt,title={Uncertainty Quality of {VGGT}: An Analysis on the {DTU} Benchmark Dataset},author={Langend{\"o}rfer, Alexander and Landgraf, Steven and Ulrich, Markus},journal={arXiv preprint arXiv:2606.16479},year={2026}}
```
Compiled in the sandbox copy: 0 errors, 15 pages, all refs/cites resolve. Placeholders (`--`): the two
beyond-window DTU rows (block W). The ETH3D beyond-window re-measure row is from v124 defaults; block X
re-runs it with the current defaults (symmetric reference, corroborated-only) — replace if it moves.

## 4. Remaining runs

```bash
cd ~/smr && unzip -o smr_updates129.zip && source .venv/bin/activate && python tests/test_dtu_eval_icp.py | tail -1
SS=~/data/dtu/SampleSet/MVS\ Data; PD=~/data/dtu/Points/stl
```

### W. DTU beyond-window rows with the all-view frame (GPU 2, ~2.5 h) — P32
```bash
L=outputs/reports/dtu_windows_v129.log
for f in data/gt/dtu/scan*.npz; do s=$(basename $f .npz | tr -d scan)
  for v in raw ca; do [ $v = ca ] && CA="--content-align --fallback none" || CA="--fallback none"
    CUDA_VISIBLE_DEVICES=2 python experiments/points_suite.py --gt $f --backbone vggt --w 16 --overlap 8 --k-ctx 0 --stride 1 \
      --points-from pointhead --conf-abs 2.0 $CA --out-dir outputs/points/dtu_win129_${v}_s$s 2>&1 | grep -E "symmetric|fused:" | tee -a $L
    echo "== win_$v scan$s ==" | tee -a $L
    python scripts/dtu_eval.py --pred outputs/points/dtu_win129_${v}_s$s/fused.ply --scan $s --sampleset "$SS" --points-dir $PD --icp 50 | tee -a $L
  done
done
python scripts/dtu_summary.py $L
```
**P32:** rejections ≤ 1 per row; `win_raw` mean Overall 0.65–0.85; `win_ca` ≤ **0.60**, better on ≥ 20/23.
Fill the two `--` rows of `tab:dtu` with the standard-22 means.

### X. ETH3D beyond-window re-measure row with current defaults (GPU 0, ~30 min) — P33
```bash
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt $f --backbone vggt --w 16 --overlap 8 --k-ctx 0 --stride 1 \
    --content-align --fallback none --save-maps --out-dir outputs/points/eth_win129_${sc} 2>&1 | grep -E "symmetric|fused:" | tee -a outputs/reports/eth_windows129.log
  echo "== $sc ==" | tee -a outputs/reports/eth_windows129.log
  python scripts/eth3d_pointmap_eval.py --maps outputs/points/eth_win129_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc --gt official --source fused \
    --json outputs/reports/eth_windows129.jsonl | grep "^umeyama " | tee -a outputs/reports/eth_windows129.log
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_windows129.jsonl
```
**P33:** 13-scene mean Overall within ±0.03 of 0.424 (raw row 0.467 unchanged).

## 5. After W and X
Both tables complete → Overleaf; then the streaming backbones (StreamVGGT, STream3R raw vs +\smr{}) on
DTU/ETH3D as the row where a memory belongs natively; then Mengmi's compute table for the read
(4 passes ≈ 4× inference time, memory unchanged — worth stating next to the numbers).
