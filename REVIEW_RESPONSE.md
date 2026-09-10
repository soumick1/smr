# Response to the third-party review (received 2026-09-08)

Status legend: **DONE** (in the paper / code), **IN PROGRESS**, **PLANNED** (with cost), **NOT DOING** (with reason),
**USER** (needs a fact only Soumick can check). Numbers cite the runs that produced them; every "done" item points to
where it lives. Updated as items close.

---

## 1. "Prove that the scaffold matters" (flat key-value baseline; decomposition) — **DONE, outcome: a tie**

Run: `pilot_a --index {flat,template,dynamics}` — identical cue, top-5 proposals, two-site verification, anchored
re-measurement, online correction and PGO; only the memory that ranks candidates differs. Cached passes.

| arm | 7-Scenes seq-01 ATE raw→+SMR→+PGO (closures) | CO3D AUC@30 raw→+SMR→+PGO | full test split (18 traj.) +SMR / +PGO |
|---|---|---|---|
| template (scaffold as run) | 0.045 → 0.030 → 0.029 (4.1) | 88.8 → 93.4 → 93.1 | 0.042 / 0.038 |
| flat key–value (cosine) | 0.045 → 0.033 → 0.033 (3.4) | 88.8 → 93.3 → 92.9 (scaffold better on 6/12 orbits) | 0.040 / 0.039 (scaffold better on 8/18, worse on 9/18) |
| dynamics (settled bumps) | 0.045 → 0.096 → 0.094 (3.6); redkitchen 0.488 | 88.8 → 93.1 → 93.0 | — |

Conclusion written into App. L (first block of Table 8 + paragraph "Does the scaffold's index matter for stitching?"):
the long-range gains of Tables 1–2 come from recognise → re-measure → verify → revise around the frozen model, and any
content-addressable memory that proposes the same candidates obtains them. The scaffold's contribution is measured where
it acts (capacity/aliasing, coherent–incoherent division of labour, cross-session persistence, pose-consistent rendering
recall) — not by the ranking of a closure proposal. The dynamics/template equivalence (synthetic) does not transfer to
one real room; stated.

Consequence for the framing (Sec. 1, Sec. 7): the paper's claim is "an external, revisable memory around frozen models";
it no longer implies the scaffold is *necessary* for the stitching numbers. **PLANNED**: one sentence in Sec. 1 and the
conclusion to make this explicit (part of the wording pass, item 6).

## 2. W_hs / W_sh, the content card, the 64-D bank index — **Figure DONE (Soumick); definitions PLANNED**

Code (`memory/hetero.py`, `stitch/memory_index.py`, `pipeline/bind.py`): W_hs ∈ ℝ^{448×1024} and W_sh ∈ ℝ^{1024×448}
exist (RLS, λ=1e2); s is the 448-d cue **only**; there is no 64-d index code — the bank index is the card's integer row.
Address h = k-WTA(W_gh g); write = RLS update of (h, s); retrieval h̃ = W_sh s, rank stored H by overlap, cosine gate,
geometric verification. Figure 1 corrected by Soumick. **PLANNED**: the four-line definition in Sec. 3 / App. C.3
(half a day, text only).

## 3. Descriptor: pooled backbone features vs DINOv2 vs engineered cue — **DONE; recommendation: keep DINOv2**

Ablation on one backbone and identical passes (Table 10): DINOv2 4.0 closures/seq, 7-Scenes 0.060, CO3D 93.4;
pooled features 2.9, 0.062, 91.2; engineered cue 0.7, 0.061, 91.4. Features propose more than the engineered cue but
convert to no more accuracy, and tie the bank to one backbone. Every table in the paper was produced with DINOv2.
Recommendation (and what my Overleaf says): DINOv2 is the scaffold's descriptor; features are the measured alternative.
This dissolves the review's four downstream requests (rerun under the new descriptor; per-backbone feature table; PCA
"N views" placeholder; weakening of "backbone-agnostic") — they only arise if features are adopted. **USER**: the second
Overleaf currently describes the feature cue as the method; decide, and I align the text either way.

## 4. Hidden compute issue (O(N) descriptor passes) — **PLANNED (audit), moot if DINOv2 stays**

With DINOv2 the per-keyframe cost is one ViT-S forward (~5 ms); it must still be listed in the compute appendix. With
pooled backbone features it is one single-frame backbone pass per keyframe (~0.1 s for VGGT; 200 frames ≈ 20 s per
sequence) and every timing/GFLOPs number must include it. **PLANNED**: add the descriptor line to App. G either way.

## 5. 7-Scenes protocol — **DONE (run); USER (which protocol the baselines use)**

Table 2 was seq-01 per scene. Ran the dataset's standard test split (18 trajectories) at the Table-2 configuration
(after fixing a pass-cache collision bug that had silently reused seq-01 passes for other sequences — v171):
raw 0.049 → +SMR 0.042 → +SMR+PGO 0.038 (per-scene mean of trajectory ATE). Per trajectory: +SMR improves 9/18, worsens
9/18 (large errors fall a lot; several good chains lose 1–3 cm; PGO recovers most). Best average of the table under both
protocols; +SMR alone ties SLAM-Former (0.042). Table 2 now shows both blocks; Sec. 5.3, abstract and conclusion say
"best average", not "outperforms". **USER**: confirm whether MASt3R-SLAM / VGGT-SLAM report seq-01 or the full split;
that block goes first.

Multi-session (StreamVGGT 0.293→0.064, STream3R 0.337→0.086): **PLANNED** — exact session construction, reset point,
what stays in the bank, evaluation protocol, plus the persistence timeline plot (item 9.7). Needs one re-run with
`--save-est` (~30 min).

## 6. Pass-through / "bit-identical" / "without training" / "backbone-agnostic" / "train end-to-end" — **PLANNED (½ day)**

All correct as criticisms. Rewrites: the trajectory write-back path preserves within-window relative camera geometry;
the content-level read deliberately revises dense geometry. "Without backbone fine-tuning or task-specific gradient
training" (W_hg is ridge-fitted). "Implementation and interface are backbone-generic" (bank is backbone-independent
only with DINOv2 cues). Evaluation design: the NVS decoder is trained on raw geometry and frozen; contribution (4)
likewise. PGO: optional refinement where the online estimate is good, needed on the full split — say both.

## 7. NVS / dense read as test-time ensembling — **PARTLY DONE; DTU baseline PLANNED**

Agreed on the narrower claim ("repeated writes to a common address yield a more useful consensus geometry"); NVS is not
used as evidence for the scaffold. Table 9 already shows the read without re-measure ties the read on NVS (known
cameras). **PLANNED**: matched naive four-pass ensemble without SMR alignment on DTU (22 scans × VGGT, ~1 h), where the
re-measure should matter because cameras come from the backbone. K=8 result (VGGT +1.10 dB, STream3R −1.52) is in
Table 9; Table 3 keeps the a-priori K=4.

## 8. Native full-context upper bound — **PLANNED (cheap)**

`pilot_a` already records a `ceiling` (single full-N pass) in every report where it fits; VGGT / VGGT-Ω / Fast3R at
N=200 can be read off and added to the sequence-length curves (`fig_results.tex`, currently commented out).

## 9. Related work 2026 (TALO, AMB3R, GlueMap, EAGLE) — **PLANNED (text); running TALO: NOT DOING**

Position SMR against submap alignment (TALO, GlueMap) and training-free pipelines (AMB3R): persistent revisable episodic
storage, recognition, re-measurement, verification, retrospective write-back, cross-session persistence. Running TALO
on a shared setup is not feasible before the deadline; one honest paragraph instead. Bib entries to be verified from the
PDFs.

## 10. Plots — **IN PROGRESS: tooling done (v173), data run pending (`bash scripts/mech_data.sh`)**
All nine as `scripts/fig_mechanism.py` subcommands; `scripts/mech_data.sh` produces the saved trajectories/candidate logs and
renders everything (~45 min GPU). Settling curves already made from the real scaffold code (CPU): incoherent states are
detected (novelty drops 2 orders) but only partly corrected (0.20 -> 0.08 m at one step, then wanders); coherent shifts are
invisible. The appendix sentence 'transverse perturbations contract' must say 'are detected; correction is the landmark
re-anchor' -- the code's own docstring already does.

1. error vs window separation (from the pairwise matrices; 37 orbits; ~15 min GPU) — feasible
2. Δ vs revisit density, CO3D + RE10K on one plot (from reports; no inference) — feasible
3. settling experiment (valid / incoherent / coherent, iteration on x; tier-1 code, CPU) — feasible, moderate
4. retrieval → verification funnel (from event logs; recall@K needs GT revisit pairs) — feasible, moderate
5. flat vs scaffold bars — data exists (item 1)
6. sequence-length curves with native points — Table 1 + ceilings; feasible
7. multi-session timeline with reset line — needs the `--save-est` re-run (item 5)
8. NVS ΔPSNR histogram with the three qualitative examples marked (their +0.41/+1.14/+1.16 vs the +0.24 mean) — from
   the jsonl; feasible and important against the cherry-picking reading
GPU-memory-vs-N line plot from Table 8 — after the above.

## 11. Small corrections — **PLANNED (½ day)**

Abstract: 22 scans and 0.441→0.396 stated. DTU: "one pass, no cross-window graph" (not "49 views fit a 32-view
window"). 9.3× gating: include the tier-1 table or drop the number. Content bank: "optional task-specific content", not
"per-primitive appearance features used by rendering". ≈19K objects (18,947 usable), 174 held-out. Table 6 bottom block
wording. PGO wording (item 6).

## 12. Strongest framing — **AGREED**

"Modern geometry models measure local geometry well; long-horizon failure is a memory problem; SMR supplies an external
revisable memory: appearance retrieves, the backbone re-measures, geometry verifies, accepted evidence rewrites the past;
the scaffold keeps the memory state stable while the geometric loop handles coherent drift." After item 1 the last
clause is supported by the memory experiments, not by the stitching tables, and the text will say which is which.

## 13. GlueMap as an offline reference row — **RECOMMENDED, not started**
Offline global SfM around a feed-forward model (github.com/colmap/gluemap): the strongest alternative answer to "what
recovers what windowing loses". Add to Tables 1 (CO3D/RE10K N=200) and 2 (7-Scenes, same 200 keyframes) labelled
"offline global SfM"; it is not a loop-closure competitor and cannot appear in streaming/multi-session rows. ~1 day.
GaussianGPT (ECCV 2026): one related-work sentence (generation, not reconstruction); not a baseline.
arXiv 2609.04026 (holistic BA): not relevant.

---

### Scoreboard of predictions made while answering the review
P66 half · P67 hit · P68 hit (flat ≈ scaffold) · P69 half (full split +40 %; pumpkin's share falls) — running total
29.5 / 65 across the project.
