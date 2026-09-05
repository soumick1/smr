# smr_updates138 — downstream dense benchmarks: DONE. Final Table 1.

## 1. P42 hit; the gauge is consistent
VGGT (22 scans) raw 0.551/0.332/0.441, read 0.476/0.316/0.396 — identical to the pre-rebuild numbers; scan13 0.288,
scan1 0.291. Every cell that moved vs v136 moved down (degenerate candidates gone: DUSt3R scan13 1.25→0.79, VGGT-Ω
scan13 1.23→0.68, Fast3R scan49 8.91→3.02); no cell worsened by more than 0.05. Remaining rejections are the backbones'
own gross failures (Fast3R 4–6, StreamVGGT 1–5, DUSt3R 2).

## 2. Final dense columns (22 DTU scans, mm; 13 ETH3D scenes, m; Overall)
| backbone | DTU raw → +SMR read | ETH3D raw → +SMR read |
|---|---|---|
| DUSt3R (swin-5 on DTU) | 2.083 = | 0.742 = |
| Fast3R | 4.403 → 4.077 | 0.780 → 2.287 (read hurts: non-commensurable orderings) |
| MASt3R | 0.799 = | 0.653 = |
| VGGT | 0.441 → **0.396** | 0.490 → **0.437** |
| VGGT-Ω § | 0.749 → 0.655 | 0.047 → 0.043 (§ contamination caveat) |
| π³ | 1.041 = | 0.122 = |
| StreamVGGT native / windows raw → re-measure | 2.889 → 2.253 (read) ; 3.356 → 3.296 | 0.707 → 1.533 ; 0.974 → 1.006 |
| STream3R native / windows raw → re-measure | 1.714 → 1.217 (read) ; 1.743 → 1.879 | 0.596 → 0.674 ; 0.771 → 0.946 |
"=" : order-invariant backbones, read = identity by construction.

## 3. Deliverables
- `iclr2027_overleaf_v138.zip`: the Overleaf project with the final merged Table 1 (`tab_main.tex`, stamped),
  `downstream.tex` (prose), `downstream_appendix.tex` (policy ablation), the bib entry; compiles 0 errors / 0 overfull.
- Code: `smr_updates138.zip` (all scripts, tests, notes). Nothing further runs for this section.

## 4. Scoreboard: 17.0/42 (P41 half, P42 hit). Next: NVS training on GSO.
