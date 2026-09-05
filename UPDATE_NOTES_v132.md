# smr_updates132 — downstream section closed: tables, captions and appendix final

## 1. Scored (scoreboard 15.0/38)
- **Z1 / P35 half.** ETH3D windows, median fusion: all pixels kept on every scene (n_corr = single-pass count);
  raw 0.530/0.404/0.467 → re-measure **0.497/0.371/0.434** (mean −7 %; 6/13 better, pred. ≥ 8; delivery_area
  1.14→0.79, facade 1.16→0.96, electro 0.19→0.12, office 0.043→0.024).
- **Z2 / P36 half.** DTU windows, median fusion, standard 22: raw 1.907/0.590/1.249 → **1.529/0.486/1.008**, better
  on **20/22** (pred. ≥ 18 ✓; means above the predicted 1.20/0.85). Six scans (4, 10, 23, 29, 33, 34) unlock in one
  variant; the 16 locked: 0.944 → **0.650**, 16/16 better.
- Policy asymmetry, now measured on both datasets: dropping disagreeing overlap pixels helps DTU (0.906 vs 1.008;
  junction errors) and hurts ETH3D (0.562 vs 0.434; depth-scale differences). The table uses the median (the
  natural two-witness consensus, all pixels kept, same on both datasets); the alternative is the appendix ablation.

## 2. Paper (final for this section)
- `paper/downstream.tex`: `sec:mvs`, `tab:dtu` (all rows), `tab:eth3d` (all rows), captions state the read
  (corroborated-only) and the fusion (median) policies and point to the appendix.
- `paper/downstream_appendix.tex`: `app:policies` + `tab:policies` — every alternative policy with its number
  (read: corroborated 0.437 / median 0.473 / abstain 0.487 / earliest 0.484; fusion: DTU 1.249→1.008 median vs
  1.280→0.906 corroborated; ETH3D 0.467→0.434 vs 0.562). Hook: `\input{downstream_appendix}` at the end of
  `appendix.tex`.
- Compiles in the sandbox Overleaf copy: 0 errors, 0 undefined references, 15 pages.

## 3. Final table rows
| DTU (mm, 22 scans) | Acc | Comp | Overall |
|---|---|---|---|
| published VGGT (theirs) | 0.389 | 0.374 | 0.382 |
| KIT feed-forward VGGT-p / VGGT-d (15 scenes) | 0.74 / 1.53 | 0.44 / 0.40 | 0.59 / 0.97 |
| ours VGGT-d (GT cams) | 1.064 | 0.375 | 0.720 |
| ours VGGT-p | 0.552 | 0.330 | 0.441 |
| **ours VGGT-p + SMR read** | **0.476** | **0.315** | **0.396** (23/23 scans better) |
| windows raw → + re-measure | 1.907 → 1.529 | 0.590 → 0.486 | 1.249 → 1.008 (20/22); locked 16: 0.944 → 0.650 |

| ETH3D (m, 13 scenes) | Acc | Comp | Overall |
|---|---|---|---|
| published VGGT depth+cam | 0.873 | 0.482 | 0.677 |
| ours protocol row (masked / all predictions) | 0.415 / 0.686 | 0.565 / 0.553 | 0.490 / 0.619 |
| **ours + SMR read (corroborated)** | **0.274** | 0.599 | **0.437** |
| windows raw → + re-measure | 0.530 → 0.497 | 0.404 → 0.371 | 0.467 → 0.434 |

## 4. Overleaf insertion (3 edits)
1. Replace `\emph{Remaining tasks (...) follow in the results revision.}` in main.tex with `\input{downstream}`.
2. Append `\input{downstream_appendix}` to appendix.tex.
3. Add to references.bib:
   `@article{langendoerfer2026vggt,title={Uncertainty Quality of {VGGT}: An Analysis on the {DTU} Benchmark Dataset},author={Langend{\"o}rfer, Alexander and Landgraf, Steven and Ulrich, Markus},journal={arXiv preprint arXiv:2606.16479},year={2026}}`

## 5. Next
Streaming backbones (StreamVGGT, STream3R) raw vs +SMR on DTU/ETH3D — the row where a memory belongs natively;
compute line for the 4-ordering read (4× inference, memory unchanged) for Mengmi's table.
