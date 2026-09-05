# smr_updates137 — table fully populated; one last CPU re-score for the DTU column; GPUs are free

## 1. v136 re-score scored (P41 half)
Rejections fell from 8–18 per row to 0–3 (Fast3R 7–8 remain: its 49-view clouds arrive ~20 % off and 130 mm away).
Rows (DTU Overall, mm): MASt3R 0.802; VGGT-Ω 0.775 → 0.682 (read); π³ 1.041; STream3R 1.746 → 1.218; StreamVGGT
2.924 → 2.297; DUSt3R (swin-5, confirmed on 22/22) 2.105; Fast3R 4.724 → 4.422; VGGT 0.478 → 0.403 (21 scans:
scan1's seed evaluation failed transiently). ETH3D unchanged (final).
**Finding**: VGGT raw scan13 0.288 → 0.914 — the gauge chose a candidate that had SHRUNK the cloud by 23 %
(init scale 0.81, |t| 129 mm) and won by 0.36 vs ≈0.4 mm: medians in both directions are too forgiving for a cloud that
overlaps the densest part of the object with the wrong shape.

## 2. v137 evaluator (`scripts/dtu_eval.py`)
Fit criterion = the **truncated Chamfer mean** on subsamples (in-region pred→scan and scan→pred means with the
protocol's 20 mm cut): the objective the gauge should minimise. Regression test: a 23 %-shrunk copy scores 11.9 mm vs
0.33 mm true (36×); slid copy 12.3 mm. All other evaluator tests unchanged (equivalence with v129 on the synthetic scan).
`scripts/rescore_matrix_dtu.sh` now includes VGGT through its seed directories and archives previous logs with a stamp.

## 3. Final DTU re-score (CPU, ~1.5 h, no GPU — start NVS on the GPUs in parallel)
```bash
cd ~/smr && unzip -o smr_updates137.zip && source .venv/bin/activate && python tests/test_dtu_eval_icp.py | tail -1
screen -dmS rescore bash -c 'bash scripts/rescore_matrix_dtu.sh 2>&1 | tee outputs/reports/matrix_rescore_v137.log'
# when done:
python scripts/build_main_table.py --matrix outputs/reports/matrix --out paper/tab_main.tex
tar czf outputs/reports/matrix_results_v137.tgz outputs/reports/matrix/*.log outputs/reports/matrix/*.jsonl paper/tab_main.tex
```
**P42:** VGGT 22 scans, raw ≈ 0.44–0.48 (scan13 ≈ 0.29), read ≈ 0.40; no row worse than under v136 by more than 0.05
except where a degenerate candidate had been chosen; rejections unchanged (Fast3R's are genuine).

## 4. Overleaf zip (this round, provisional)
`iclr2027_overleaf_v137.zip` carries the merged Table 1 built from the v136 re-score: ETH3D cells final; DTU cells
provisional (they move where the gauge had picked a degenerate candidate — VGGT scan13 for sure). Stamped in the
`.tex` comment. The final zip follows the re-score in §3.

## 5. Status of the downstream work
GPU work for the dense table is complete. Remaining: the §3 CPU pass, then swapping `tab_main.tex` in Overleaf.
NVS training on GSO can start now; the three GPUs are idle.
