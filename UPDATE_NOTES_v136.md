# smr_updates136 — matrix results read; the gauge refused the other backbones; fix + CPU re-score

## 1. What the results say (v134 gauge) and why the DTU column is not yet real
- VGGT (re-scored seed): raw 0.556/0.336/0.446; read 0.484/0.373/0.429 — but scan13 0.255 → 0.85 for the same file:
  the best-of-candidates gauge picked a *slid* alignment (pred→scan median 0.52 mm vs 2.93, Comp 1.33). The fit
  criterion was one-sided.
- Every other backbone: rejection counts 8–18 of 22, Acc 2–7 mm = unaligned clouds. The rejected refinements needed
  scale changes of 6–25 % (median per backbone; DUSt3R/Fast3R ≈ 22–25 %) and moves of 20–55 mm — legitimate global
  corrections that v134's caps (5 %, 30 mm, tuned on VGGT) refused. The true runaways (scale ×0.24, moves of
  700–900 mm) are also there and must still be refused — by fit, not by magnitude.
- ETH3D (Umeyama on correspondences, unaffected): VGGT 0.490 → 0.437; **π³ 0.122** (read = identity; no failure
  scenes); DUSt3R 0.742 / MASt3R 0.653 (published 1.005 / 0.826); STream3R 0.596 → 0.674; **the read hurts
  Fast3R (0.780 → 2.29) and StreamVGGT (0.707 → 1.53)**: for reference-frame-dependent and causal models the
  orderings are not commensurable and the corroborated read keeps too little. Real property of the read; stated.
- **VGGT-Ω ETH3D 0.047** (uniform, incl. meadow): the wrapper's own docstring quotes the authors' README —
  possible benchmark contamination of the released checkpoint; ETH3D's training scenes are this benchmark.
  Its rows carry $^{\S}$ and the caption sentence. Not a bug of ours.
- DUSt3R DTU ran through the ladder (results present, 16/22 rejected under v134's caps). Confirm the rung used:
  `cat outputs/points/matrix/dust3r_dtu_s*/GRAPH | sort | uniq -c` (caption says window 5).

## 2. Evaluator v136 (`scripts/dtu_eval.py`; tests: `tests/test_dtu_eval_icp.py`, equivalence to v129 preserved)
- **Symmetric fit criterion**: mean of the median pred→scan distance (in-region points) and the median scan→pred
  distance (scan above the plane). A 30 mm slid copy scores 13.2 mm vs 0.31 for the true alignment.
- **Robust similarity initialisation with shrinking support**: medians of all points (translation), then points
  within 60 mm and 40 mm of the scan (translation + robust scale from median radial extents). Makes no assumption
  about where the cloud arrived. A 25 %-too-large, 40 mm-off cloud is recovered to the 0.3 mm noise floor.
- Caps relaxed to sanity limits (|log scale| ≤ 0.5, move ≤ 300 mm); fit decides. An 80 mm start without the coarse
  candidate is refused or left alone, never "fixed" by a runaway.

## 3. Re-score (CPU only, 4 parallel groups, ~3–4 h; no GPU)
```bash
cd ~/smr && unzip -o smr_updates136.zip && source .venv/bin/activate && python tests/test_dtu_eval_icp.py | tail -1
screen -dmS rescore bash -c 'bash scripts/rescore_matrix_dtu.sh 2>&1 | tee outputs/reports/matrix_rescore.log'
```
Old logs are kept as `outputs/reports/matrix/<bb>_dtu.v134.log`. When done:
```bash
python scripts/build_main_table.py --matrix outputs/reports/matrix --out paper/tab_main.tex
tar czf outputs/reports/matrix_results_v136.tgz outputs/reports/matrix/*.log outputs/reports/matrix/*.jsonl paper/tab_main.tex
```
**Predictions (P41):** rejections ≤ 2 per row; VGGT rows back to ≈ 0.446 / 0.396 (scan13 ≈ 0.26); every backbone's raw
Overall < 1.5 except DUSt3R (sliding-window alignment over 49 views drifts non-rigidly; ≈ 1.5–3); π³ raw ≤ 0.8.

## 4. Overleaf
No Overleaf zip this round on purpose: the dense DTU cells in a table built now would carry numbers scored under the
defective gauge. `paper/tab_main.tex` (provisional, stamped) compiles with the new caption and the VGGT-Ω marker; the
real zip follows the re-score.
