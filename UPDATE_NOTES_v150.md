# smr_updates150 — NVS rows final; Table 3 filled; paper updated (Overleaf zip: iclr2027_overleaf_v150.zip)

## GSO, all 1,033 objects, paired raw vs read through the same head (no retraining)
| backbone | raw PSNR/SSIM/LPIPS | +SMR read | dPSNR [95% bootstrap CI] | better on |
|---|---|---|---|---|
| VGGT-Ω | 18.89 / 0.796 / 0.204 | 19.07 / 0.799 / 0.200 | +0.18 [+0.09, +0.26] | 57 % |
| VGGT | 20.71 / 0.821 / 0.152 | 20.95 / 0.823 / 0.143 | +0.24 [+0.19, +0.29] | 69 % |
| π³ | 21.91 / 0.844 / 0.132 | 21.91 / 0.844 / 0.132 | +0.00 (identity, measured) | — |
| STream3R | 18.79 / 0.784 / 0.209 | 18.60 / 0.781 / 0.222 | −0.18 [−0.30, −0.07]; median +0.04 | 52 % |
Zero crashes, zero error records. The story is Table 1's: the read improves the order-dependent transformers, is the
identity for the permutation-equivariant model, and does not help the causal one (a minority of non-commensurable
objects drags STream3R's mean while the median object is unchanged).

## Scoreboard
P54 hit (π³ identity), P55 hit (VGGT +0.24 in [+0.15,+0.5]), P56 hit (VGGT-Ω read ≥ raw−0.1; it is +0.18), P57 hit
(STream3R read ≤ raw by ≤ 1 dB), P58 hit (no crashes); earlier: P48 hit (π³ raw ≥ VGGT raw), P52 miss (VGGT-Ω raw <
VGGT raw), P47 miss (absolute VGGT raw 20.7, not 24-27: the head is LGM-class), P43 lower edge (+0.24 vs +0.3-1.0
predicted → half). Running total 22.5/54.

## Paper (v150)
Table 3 filled (8 rows, bold = better of each pair; caption: one window has no junction graph, so no +SMR+PGO row).
Sec. 5.4 results paragraph written (absolute level = method class, LGM's row; read gains with CIs; π³ identity;
STream3R mean/median). Abstract: "+0.24 dB through a fixed rendering head". Limitations condensed and updated.
Compiles: 0 errors / 0 overfull; main text incl. both statements = 9 pages, references from page 10.

## Open decision (unchanged): the absolute NVS level
Parity with VGGT-NVS's 30.4 is not reachable with a Gaussian head over frozen geometry (see v148). A 2D refinement
network after the splat (the UNet idea) would likely lift all rows to 26-28 dB in 2-3 GPU-days and keeps the raw/read
comparison fair; it changes no claim of the paper. Your call; the paper as it stands is complete and honest.

## Remaining before submission (from v143): bib placeholders "verify entry" (SLAM-Former, ViSTA-SLAM, VGGT-SLAM++,
STream3R authors); companion-submission framing in Sec. 5.3 (Mengmi); Prop. 1 wording; optional Fig. 3 (qualitative NVS).
