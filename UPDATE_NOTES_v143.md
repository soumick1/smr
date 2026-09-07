# smr_updates143 — full audit and rewrite of the ICLR draft (Overleaf zip: iclr2027_overleaf_v143.zip)

## What changed (every section touched; compiles 0 errors / 0 overfull; main text incl. statements = 9 pages)
**Structure.** Abstract rewritten (concrete numbers; no "six tasks"/"provably"). Intro tightened, contributions = 4,
points at Fig. 1. Sec. 2 tightened, related-work pointer. Sec. 3 unchanged in content, tightened; the
coherent/incoherent split is now **Proposition 1** (division of labor) with an argument in App. C — stated as a design
rule, not a capacity theorem. Old Sec. 4 ("What the memory buys", which still listed two-view matching and the old
NVS design) replaced by **Evaluation design**: the four evaluations actually reported and what each exercises.
Sec. 5.1 pose (refers to Fig. 2; compute tables moved to App. G with one sentence in the text). **Sec. 5.2 dense MVS /
point maps rewritten against the final numbers** (VGGT 0.441→0.396 on 23/23 scans; ETH3D 0.490→0.437; VGGT-Ω, streaming
gains; identity rows; the read failing on Fast3R/causal models; the VGGT-Ω contamination caveat). Sec. 5.3 SLAM
tightened + one sentence on the companion submission. **New Sec. 5.4 NVS** with the empty **Table 3** (published anchors
on top, our rows below, `--` cells) and a commented Fig. 3 placeholder. **New Sec. 6 Limitations, Sec. 7 Conclusion,
Reproducibility statement, Ethics statement.**
**Figures.** Fig. 1 system overview (TikZ, no external assets): window chain with seams → scaffold (rings, modules,
grid code, W_gh/W_hg, cards) → read loop (propose, anchored pass, consensus, re-clamp). Fig. 2 (pgfplots from the
real cells): (a) CO3D AUC vs N raw/+SMR for 4 backbones, (b) RE10K control, (c) peak GPU vs N with the 46 GB wall.
**Appendix.** New: G compute & memory tables (moved), H NVS protocol/renderer/head, I related work (feed-forward
geometry, SLAM on foundation models, scaffold memory, feed-forward NVS), Prop. 1 argument in C. Existing sections kept.
**Bibliography.** Added lvsm, lgm, gso, objaverse, kerbl3dgs, splatt3r, lpips.

## Decisions I made that you or Mengmi may overrule
1. Companion-paper sentence in Sec. 5.3 ("A companion submission develops the SLAM system itself; here it serves as
   the memory's online test") — the dual-submission framing must be confirmed.
2. Proposition 1 is worded as an operational statement with an argument, not a theorem; if Mengmi prefers, demote to
   a boxed "Design rule" (one-word change) or promote with a formal proof.
3. The reproducibility/ethics statements sit on page 9 after the conclusion; ICLR does not count them toward the
   limit as far as I know — verify in the ICLR 2027 CFP. If they count, cut ~10 lines (Sec. 2 or 5.1 have slack).
4. Compute tables live in App. G; the main text keeps one sentence + Fig. 2c.

## To do before submission (flagged, not fixable from here)
- `references.bib`: four placeholder entries marked "verify entry" (SLAM-Former, ViSTA-SLAM, VGGT-SLAM++ as
  "Anonymous"; STream3R author list). Take them from the companion paper's bib.
- Fill Table 3 (NVS) from `experiments/eval_gso.py`; write the three result sentences at the `%` marker in Sec. 5.4;
  optionally uncomment Fig. 3 with a qualitative strip (inputs | raw | +SMR | GT).
- After the NVS numbers: one sentence in the abstract if the read gain is real (P43).
- Multi-session SLAM numbers appear in text only (companion table); fine for scope, but if reviewers ask, App. G
  has room for a two-row table.

## Files (Overleaf zip): main.tex, appendix.tex, fig_system.tex, fig_results.tex, tab_main.tex, tab_nvs.tex,
tab_pose_camera.tex, downstream_appendix.tex, references.bib, style files, main.pdf. `downstream.tex` is no longer
input (its content lives in Sec. 5.2) — remove it from the Overleaf project.
