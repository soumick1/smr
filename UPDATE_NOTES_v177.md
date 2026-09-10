# smr_updates177 — trust and attribution first (plan Tasks 1, 2, 5, 8; groundwork for 3, 4, 6, 7)

Built on the v167 tree from GitHub (`4d364a4`, 2026-09-08). Everything here is a NEW file except one patch to
`experiments/pilot_a.py`, applied by `scripts/apply_v177_patch.py`, which asserts its four anchors and refuses to
write if `pilot_a.py` changed there after v167. v168–v176 are not on GitHub; push them (or upload the zips) before
anything in v178 touches `anchored.py`. Numbering: v177 code; no Overleaf zip this round (text consequences listed
in §3 so they can be written from the code, as the plan asks).

## 1. What is in the zip

| file | plan task | tested here |
|---|---|---|
| `tests/test_guarantees.py` | 2 | 9 tests pass (`pytest -s` prints the measured findings) |
| `scripts/closure_reliability.py` | 5 | synthetic report + GT: funnel, recall, precision, misses, budget stats, AUC_within |
| `scripts/gate_table.py`, `scripts/gate_sweep.sh` | 5 | table on synthetic; sweep = 17 variants x (7 + 12) sequences, cached passes |
| `scripts/paired_stats.py` | 8 | reports / rows / nvs / csv modes, paired bootstrap, sign, Wilcoxon |
| `scripts/run_manifest.py`, `src/smr/utils/provenance.py`, `scripts/apply_v177_patch.py` | 1 | manifest + `--diff`; provenance lands in the report; patch idempotent |
| `scripts/seed_sweep.sh`, `scripts/compare_closures.py` | 3 | compare tested; sweep needs `--index` (v170+, not in my tree) |
| `scripts/multisession_eval.py` | 7 | two-session synthetic: joint vs per-session ATE vs frame recovery |

Patch adds to `pilot_a`: `report["provenance"]` (commit, dirty, host, torch/cuda, argv, args, checkpoint sha256 of
every `*.pt` under `third_party/checkpoints` and `checkpoints/`, hashed once and cached), and flags
`--site-rot --site-dir --extent-factor --budget-rot --budget-pos --budget-logscale --tight-rot --tight-pos
--no-smooth-junctions --top-proposals`. Defaults reproduce every existing run (verified: identical synthetic rows).

## 2. What the tests established (code truth for the text)

A. **Estimator** (`sim3.fit_poses`): rotation from orthogonal Procrustes on the frame ORIENTATIONS, scale from the
   ratio of centre spreads, translation from centroids. A two-view site determines rotation exactly; one view or
   coincident centres leave only scale unresolved and flagged (`scale_ok=False`, s=1). App. A describes a
   centres-only Horn/Umeyama fit with orientation as a check: that is not the code. AUTHOR_CHECK resolved.
B. **"Not applied online" is not an identity control.** A window with proposed sites takes its local geometry from
   the enlarged anchored pass (`pass_idx = window + anchors`), so the trajectory diverges from the raw chain at the
   first window with a proposal, accepted or not. Rejected proposals therefore change geometry state, not only the
   log. This is also why a run with ZERO accepted closures differs from raw (`--budget-rot 2,0.5,8` on synthetic:
   0 loops, ATE 0.369 vs chain 0.293). Consequence: Table 6's `corr_none` row (0.042 vs raw 0.045) mixes
   "re-measure with old frames in the pass" and "no write-back". A genuine control needs `anchored.py` to take the
   window's local geometry from the plain pass and use the anchored pass only for the closure measurement (v178).
C. **Owner-window preservation** holds exactly (residual 2e-6) for every window NOT adjacent to an accepted
   closure. At an accepted closure the default `smooth_junctions=True` blends the PRECEDING window's overlap frames
   per frame between the two placements (anti-step; TUM fr1_room AUC_in 80.8 -> 75.8 without it). With
   `--no-smooth-junctions` every window is exact. App. B's "every frame of a window receives the same similarity"
   holds away from closures; state the blend. `nosmooth` is a gate_sweep variant so its cost on real data is measured.
D. **Causality**: K windows vs the first K of a longer run (same frame list and schedule): identical proposals and
   acceptance decisions; identical placements for windows never inside a later closure span; later closures revise
   the rest (that is the point). The rgb adaptive gate uses consecutively BOUND views only (causal); DINO uses 0.5.
E. **Address convention**: a scaffold address is written at bind time and never re-addressed; `update_pose` moves
   the pose annotation and state row only. Cue retrieval is unchanged by corrections; pose-proximity proposals use
   the corrected poses. Write this convention into Sec. 4.3 / App. C.4.
F. **Operative gate constants** (all stitching tables; `test_gate_constants_are_the_paper_values`): per-site gate
   rot < 10 deg, translation direction < 25 deg, distance < 3 window spreads; robust fit 10 deg / 0.5 spread;
   two-site drift budget rot <= min(45, 10 + 3 n), pos <= 1.0 + 0.5 n spreads, |log s| <= 0.5, n = windows since the
   last anchor; single-site budget min(15, 3 + 1 n) / 0.5 + 0.15 n; at least one appearance-proposed site;
   top-5 proposals, partner offset 3. The two-site MUTUAL agreement test (App. D Eqs. 5-6) is `--site-agree`,
   default `1e9,1e9` = off. pilot_a defaults for the index: `--N-h 2048`, `--torus-N 32`, k = N_h/16 = 128,
   position-only code N_g = 3 (32^2 + 32) = 3,168, DINO cosine gate 0.5 absolute. (48^2 / 48 / 256, 7,824, 1024,
   64 are the dynamical ScaffoldState/BlockScaffold defaults of App. C.) `test_pilot_a_defaults_documented` fails
   the day a default changes, which is the mechanism the plan asks for.

## 3. Two table rows that disagree with the recorded runs (fix before anything else)

* **Table 5, flat row**: Overleaf shows 0.035 / 0.034 m and 91.3 / 90.9 AUC; the runs recorded in
  `REVIEW_RESPONSE.md` and `HANDOVER.md` are 0.033 / 0.033 and 93.3 / 92.9 (tie; scaffold better on 6/12 orbits,
  8/18 test trajectories). Table 5's default PGO cell (93.3) also disagrees with Table 6's (93.1). The honest
  limitation sentence in `main.tex` (~line 700) is commented out. `run_manifest.py --diff outputs/ablate/7scenes/default
  outputs/ablate/7scenes/index_flat` says whether any configuration explains a difference; if it prints NOTHING,
  the recorded numbers go back and "partly preserves" becomes "preserves".
* **Table 3, pi^3 row**: Overleaf shows 22.20 -> 22.79 and 21.91 -> 22.12; the records (handover, App. H,
  `tab_ablation_read.tex`: "21.91 -> 21.91 identity", "22.19 -> 22.19") say identity, which is also what
  permutation equivariance predicts. `paired_stats.py nvs` on `outputs/nvs/pi3/gso_reads{1,4}.jsonl` settles it.

## 4. VGGT-Omega checkpoint (plan Task 1, verified against the repository on 2026-09-10)

`facebookresearch/vggt-omega/reproduction.md`: after a contamination review, a retrained checkpoint
`vggt_omega_1b_512_reproduce.pt` (Aug 2026; note the name, not `416_reproduce`) SUPERSEDES the original as the
reference for benchmark comparisons and must be run with `image_resolution=416`. Its 7-Scenes AUC@30 is 82.4 vs
83.1 for the original. Our wrapper (`src/smr/backbones/vggt_omega.py`) loads `vggt_omega_1b_512.pt` at 512 = the
original, and 7-Scenes is our Table 2 dataset with this backbone. Order of work once confirmed (step 2 below):
7-Scenes seq-01 raw/+SMR/+PGO (7 sequences, no cached passes, ~1 h), then the 18-trajectory split (~3 h), then the
Omega rows of Table 1 (~1 day) and dense; the Omega NVS head either stays tied to the original checkpoint (say so)
or is retrained on the new geometry. Never mix passes from the two checkpoints: use a new cache root
(`outputs/cache_omega_repro/`). The wrapper needs a checkpoint/resolution flag (v178, small).

## 5. Commands, in order (server)

```bash
# 0. sandbox parity (either)
git add -A && git commit -m "v168-v176" && git push            # or: tar czf smr_v176_tree.tgz experiments src scripts tests paper
# 1. unpack v177, patch, run the guarantee tests
unzip -o smr_updates177.zip && python scripts/apply_v177_patch.py --check && python scripts/apply_v177_patch.py
PYTHONPATH=src python -m pytest -q -s tests/test_guarantees.py
# 2. Task 1: checkpoint audit + manifest over every report
sha256sum third_party/checkpoints/vggt_omega_1b_512.pt; ls -la third_party/checkpoints/     # compare with the sha256 shown on huggingface.co/facebook/VGGT-Omega for both files
python scripts/run_manifest.py --glob 'outputs/**/*.json' 'outputs2/**/*.json' --csv outputs/manifest.csv --md outputs/manifest.md
python scripts/run_manifest.py --diff outputs/ablate/7scenes/default outputs/ablate/7scenes/index_flat
python scripts/run_manifest.py --diff outputs/ablate/co3d/default outputs/ablate/co3d/index_flat
grep -c '"disagree"' outputs2/reports/*.json | sort -t: -k2 -n | tail -2                        # 0 everywhere => mutual agreement test was off (as in code)
# 3. Task 5: reliability from EXISTING reports (no GPU)
python scripts/closure_reliability.py --json outputs2/reports/pilotA_7scenes_*_seq01_vggt_omega_s5_c32.json --gt-dir data/gt --md outputs/rel_7scenes_seq01.md --csv outputs/rel_7scenes_seq01.csv
python scripts/closure_reliability.py --glob 'outputs/7scenes_test/*_vggt_omega_template.json' --gt-dir data/gt --md outputs/rel_7scenes_test.md
python scripts/closure_reliability.py --glob 'outputs/ablate/co3d/default/*.json' --gt-dir data/gt/co3d_full --md outputs/rel_co3d.md
python scripts/closure_reliability.py --glob 'outputs/ablate/7scenes/sites1/*.json' --gt-dir data/gt --md outputs/rel_sites1.md     # why one site hurts
python scripts/closure_reliability.py --glob 'outputs/ablate/7scenes/index_flat/*.json' --gt-dir data/gt --md outputs/rel_flat.md
# 4. Task 8: paired statistics for the comparisons the paper makes
python scripts/paired_stats.py rows --reports 'outputs2/reports/pilotA_7scenes_*_seq01_vggt_omega_s5_c32.json' --row-a chained --row-b smr --metric ate_rmse --per-item
python scripts/paired_stats.py rows --reports 'outputs/7scenes_test/*_vggt_omega_template.json' --row-a chained --row-b smr --metric ate_rmse --per-item
python scripts/paired_stats.py reports --a 'outputs/ablate/7scenes/default/*.json' --b 'outputs/ablate/7scenes/index_flat/*.json' --row smr --metric ate_rmse --per-item
python scripts/paired_stats.py reports --a 'outputs/ablate/co3d/default/*.json' --b 'outputs/ablate/co3d/index_flat/*.json' --row smr --metric auc30 --per-item
for bb in vggt vggt_omega pi3 stream3r fast3r streamvggt; do echo == $bb; python scripts/paired_stats.py nvs --a outputs/nvs/$bb/gso_reads1.jsonl --b outputs/nvs/$bb/gso_reads4.jsonl --metric psnr; done
# 5. sweeps on cached passes (GPU 0; ~1.5 h + 15 min)
CUDA_VISIBLE_DEVICES=0 bash scripts/gate_sweep.sh && python scripts/gate_table.py --root outputs/ablate/gate --gt-dir data/gt --md outputs/gate_sweep.md
CUDA_VISIBLE_DEVICES=0 bash scripts/seed_sweep.sh
for s in 0 1 2; do python scripts/paired_stats.py reports --a 'outputs/ablate/seeds/7scenes/flat/*.json' --b "outputs/ablate/seeds/7scenes/seed$s/*.json" --row smr --metric ate_rmse; done
python scripts/compare_closures.py outputs/ablate/seeds/7scenes/seed0/pumpkin.json outputs/ablate/seeds/7scenes/flat/pumpkin.json
```
Send back: `outputs/manifest.md`, the `rel_*.md`, the paired_stats outputs (paste), `outputs/gate_sweep.md`, the
`sha256sum` line, and the two `--diff` outputs.

## 6. Predictions (scoreboard continues from P74)

P75 the server's Omega file hashes to the ORIGINAL HF checkpoint -> Table 2 rerun required.
P76 7-Scenes seq-01 pooled: closure precision at 2x tolerance >= 0.9; site recall 0.6-0.8; most misses are
    "no proposal" (DINO gate 0.5), not budget rejections.
P77 gate sweep: every 3x3 budget cell within +-0.004 m of default except the tightest corner (fewer loops, ATE up
    <= 0.006); x2 accepts >= 1 false closure on some scene; site gates within noise; `nosmooth` lowers AUC_within
    on scenes with closures and changes ATE by <= 0.002.
P78 three scaffold seeds within +-0.002 m of each other and each ties flat (sign p > 0.3).
P79 paired_stats nvs: pi^3 delta 0.00 with CI [0.00, 0.00]; VGGT GSO +0.24 dB with a CI excluding 0.

## 7. v178 (needs the current `anchored.py` / `pilot_a.py` / `eval_gso.py`)

1. `--local-from plain|anchored`: window geometry from the plain pass, anchored pass only for the closure
   measurement -> the genuine no-write-back control and a clean separation of re-measurement from revision (2B).
2. `--replay-sites FILE`: feed a saved candidate list through both indices (plan Task 3, second comparison).
3. Dense-read attribution harness (plan Task 4): one pass through the read preprocessing; four copies of one
   prediction; four orderings + alignment + median in a plain array; full read. Same cached predictions and frozen
   decoders; matched valid pixels; Gaussian counts logged. Stage-by-stage trace for pi^3.
4. Bank save/load + the four-condition multi-session runner (empty / retained / reloaded / unrelated bank), scored
   with `multisession_eval.py` (plan Task 7).
5. Stage timing incl. every anchored pass attempted, cold vs cached (plan Task 6); wrapper flag for the Omega
   reproduce checkpoint + `image_resolution=416`; `--cache-root`.
6. Text pass from §2-3: App. A, B, C.4, C.5, D, Sec. 4.1-4.2, Fig. 1, Table 11, App. J ("the anchored pass
   replaces the window's pass; the composed GFLOPs count a full extra 36-frame pass per accepted closure and are an
   upper bound"), and the two table rows of §3.
