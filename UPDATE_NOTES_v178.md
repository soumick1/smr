# smr_updates178 — fixes for the v177 run, what your outputs already settle, and the Ω-reproduce backbone

## 1. What went wrong on the server (and the fixes in this zip)

| symptom in your paste | cause | fix |
|---|---|---|
| gate_sweep: every non-default variant `unrecognized arguments: --budget-rot` | `scripts/apply_v177_patch.py` was not run before the sweep (the tests do not need it, so they passed) | run the patch (step 1 below); the 19 `default` runs are kept, the rest resume |
| 7scenes_test reliability: `chess_seq03 ... [gt=7scenes_chess_seq01.npz]`, precision 0.10 | my GT resolver matched the scene name and ignored the sequence id, so seq03…seq14 were scored against seq01's GT. **All test-split reliability numbers in your paste are invalid**; the seq-01 and CO3D numbers are fine | resolver now keys on the report file name; a `seqNN` token must match (seq03 == seq-03 == seq_3); `--debug-gt` prints candidates |
| CO3D reliability `[]` = no GT matched | resolver did not accept the `co3d_` prefix / tokenised names | token matching; `co3d_apple_110_...npz` now matches `apple_110_...json` |
| `paired_stats rows` on the test split: n=7 instead of 18 | pairing key was the `scene` field, and chess_seq03/seq05 both say "chess" | pairs by file stem (variant suffixes `_template/_flat/_seedN` stripped); `--key scene` restores the old behaviour |
| `run_manifest --diff` says NOTHING differs for default vs index_flat | pre-v177 reports do not record `--index` (or seeds, gate flags); the message was misleading | says so explicitly; from v177 on, provenance records every flag |
| `grep -c disagree` showed KITTI reports with 21–26 | those are the v75 `_gate` A/B runs (mutual agreement ON); not the Table 2 / CO3D runs | check the relevant reports (step 2) |
| seed_sweep interrupted, compare_closures file-not-found | you stopped it; no seed dirs yet | rerun after the patch |

New in v178: `closure_reliability.py` gains a co-visibility criterion (`covis`: centres within 30 % of the GT diagonal
and viewing directions within 60°) and, with `--est-dir`, a **usefulness** measure per accepted closure (cross-window
GT relative-pose error between the closed windows in the raw chain vs the row, useful/harmful at ±0.5°); `gate_sweep.sh`
now writes `--save-est` npz beside every report so usefulness is available; `paired_stats nvs` prints the `reads`
field found inside each jsonl (needed for §2.3). `src/smr/backbones/vggt_omega_repro.py` + `apply_v178_patch.py` (§4).

## 2. What your outputs already settle

### 2.1 Table 5 (scaffold vs flat index) — recorded runs say TIE, not "partly"
7-Scenes seq-01, +SMR row: scaffold 0.0305 m, flat 0.0332 m, Δ = +0.0027 [95 % paired bootstrap −0.0013, +0.0084],
improved 2/7, worsened 4/7, tie 1, sign p = 0.69. PGO: 0.0291 vs 0.0328 (from the `--diff` listing). Redkitchen alone
(0.028 vs 0.046) is the difference; on the other six scenes flat ≈ scaffold. CO3D 12 orbits, AUC@30: 93.36 vs 93.28,
Δ = −0.07 [−0.64, +0.49], 6/6, p = 1. The Overleaf's 0.035/0.034 and 91.3/90.9 are not in the reports on disk; the
recorded values are 0.033/0.033 and 93.3 (+SMR) / 92.9 (+PGO, from the handover). Table 5 and the three sentences that
say the flat memory recovers "some but not all" must change to the tie, and the commented limitation returns.

### 2.2 Table 2 (7-Scenes seq-01, raw → +SMR): the mean gain is two scenes
0.0451 → 0.0305 (Δ −0.0146 [−0.0387, +0.0008]), improved 4/7, median Δ −0.0001. Pumpkin (−0.084) and office (−0.021)
carry the average; fire, redkitchen and stairs are 1–2 mm worse; chess −3 mm; heads unchanged. With n = 7 the interval
includes zero. The per-scene table already shows this; the text should say the gain is where long revisits exist
(pumpkin, office) and not claim uniform improvement. The 18-trajectory split numbers must be recomputed with the fixed
pairing (step 3); the n = 7 line in your paste was the last sequence per scene only.

### 2.3 Table 3 (NVS, GSO, paired over 1,033 objects)
| backbone | raw → read (jsonl means) | Δ PSNR [95 % CI] | improved | Overleaf Table 3 |
|---|---|---|---|---|
| VGGT | 20.71 → 20.95 | +0.24 [+0.19, +0.29] | 716/1033 | matches |
| VGGT-Ω | 18.89 → 19.07 | +0.18 [+0.10, +0.26] | 586/1033 | matches |
| π³ | 21.91 → 21.91 | +0.001 [−0.001, +0.003] | 535/1033, p = 0.25 | **shows +0.21 (GSO) / +0.59 (held-out): unsupported; identity** |
| STream3R | 18.79 → 18.60 | −0.18 [−0.30, −0.07] | 534/1033, median +0.04 | **shows 18.60 → 18.79: reversed** |
| Fast3R | 14.07 → 14.20 | +0.14 [+0.08, +0.20] | 499/1033 (167 ties) | matches |
| StreamVGGT | 18.30 → 18.59 | +0.30 [+0.23, +0.37] | 653/1033 | matches |
STream3R: `gso_reads1.jsonl` has the higher mean. Table 8's own eight-ordering block lists raw = 18.79, which agrees with
the jsonl and contradicts Table 3's row. v178 `paired_stats nvs` prints the `reads` field inside each file; if
`gso_reads1.jsonl` indeed holds reads = 1 records, the Table 3 STream3R row is swapped and the four-ordering read hurts
STream3R in the mean (heavy-tailed: median +0.04, losses of 1.5 dB when it fails), consistent with the eight-ordering
degradation already reported. Check LPIPS/SSIM the same way (step 3).

### 2.4 Closure reliability (7-Scenes seq-01, valid GT; 71 windows)
* 76 sites proposed, 44 verified. The **only** per-site gate that fires is the extent gate (32 rejections; rotation and
  direction gates 0). The drift budget rejected **nothing**: 29 closures tested, 29 accepted. On Table 2's runs,
  verification = extent gate + "at least one appearance-proposed verified site"; the budget is inert (it fires once on
  CO3D, three times on the test split). Say so in App. D; do not describe the budget as the working gate on 7-Scenes.
* 13/29 accepted closures had two sites; scale was measured from anchors in 11/29 (fixed elsewhere).
* Strict camera-proximity precision 0.48 (2× tolerance 0.72), yet pumpkin's three closures are all "false" by proximity
  and cut ATE 0.160 → 0.077. Camera proximity is the wrong truth for a loop closure (two cameras 1 m apart viewing the
  same wall are a valid closure). v178 adds `covis` and `useful`; rerun (step 3) before quoting any precision.
* CO3D: extent gate rejects 118/140 sites; 19 closures tested, 18 accepted; median span 7 windows.
* `sites1`: precision like default (0.46) but Table 6 shows worse ATE → the one-site degradation is measurement
  quality (2 anchor frames, fixed scale in 24/26 closures), not recognition. The usefulness metric will show it.
* Anchored passes attempted = windows with proposals = 50 of 71 (all accepted-or-not); Table 9's accounting counts
  only accepted closures.

### 2.5 VGGT-Ω checkpoint
Your file: `vggt_omega_1b_512.pt`, 4,576,706,117 B (4.58 GB), dated Aug 24; HF lists the original at 4.58 GB under
that name and the retrained one under `vggt_omega_1b_512_reproduce.pt`. It is the original (P75 confirmed by name and
size; the sha256 is visible on the HF file page when logged in). Table 2 has to be rerun with the retrained checkpoint
(step 4); the Ω rows of Tables 1 and 3 follow.

## 3. Commands, in order

```bash
# 1. patch first (this is what the sweep needed), rerun tests
unzip -o smr_updates178.zip && python scripts/apply_v177_patch.py && python scripts/apply_v178_patch.py
PYTHONPATH=src python -m pytest -q tests/test_guarantees.py
# 2. did the mutual-agreement test run in any Table 2 / CO3D report? (expect all 0)
grep -c '"disagree"' outputs2/reports/pilotA_7scenes_*_seq01_vggt_omega_s5_c32.json outputs/7scenes_test/*_template.json outputs/ablate/co3d/default/*.json | awk -F: '$2>0'
# 3. redo the invalid/incomplete analyses (no GPU)
python scripts/closure_reliability.py --glob 'outputs/7scenes_test/*_vggt_omega_template.json' --gt-dir data/gt --debug-gt --md outputs/rel_7scenes_test.md 2>&1 | grep -v candidates
python scripts/closure_reliability.py --glob 'outputs/ablate/co3d/default/*.json' --gt-dir data/gt --md outputs/rel_co3d.md
python scripts/paired_stats.py rows --reports 'outputs/7scenes_test/*_vggt_omega_template.json' --row-a chained --row-b smr --metric ate_rmse --per-item
python scripts/paired_stats.py rows --reports 'outputs/7scenes_test/*_vggt_omega_template.json' --row-a chained --row-b smr_pgo --metric ate_rmse
python scripts/paired_stats.py reports --a 'outputs/7scenes_test/*_vggt_omega_template.json' --b 'outputs/7scenes_test/*_vggt_omega_flat.json' --row smr --metric ate_rmse --per-item
for m in psnr ssim lpips; do python scripts/paired_stats.py nvs --a outputs/nvs/stream3r/gso_reads1.jsonl --b outputs/nvs/stream3r/gso_reads4.jsonl --metric $m | head -4; done
ls -la outputs/nvs/stream3r/ outputs/nvs/pi3/          # file dates: were reads1/reads4 written by the runs you think?
# 4. sweeps on cached passes (GPU 0), now with --save-est; then reliability WITH usefulness
CUDA_VISIBLE_DEVICES=0 bash scripts/gate_sweep.sh
python scripts/gate_table.py --root outputs/ablate/gate --gt-dir data/gt --md outputs/gate_sweep.md
python scripts/closure_reliability.py --glob 'outputs/ablate/gate/7scenes/default/*.json' --gt-dir data/gt --est-dir outputs/ablate/gate/7scenes/default --md outputs/rel_default_useful.md
python scripts/closure_reliability.py --glob 'outputs/ablate/gate/7scenes/nosmooth/*.json' --gt-dir data/gt --est-dir outputs/ablate/gate/7scenes/nosmooth --md outputs/rel_nosmooth_useful.md
CUDA_VISIBLE_DEVICES=0 bash scripts/seed_sweep.sh
for s in 0 1 2; do python scripts/paired_stats.py reports --a 'outputs/ablate/seeds/7scenes/flat/*.json' --b "outputs/ablate/seeds/7scenes/seed$s/*.json" --row smr --metric ate_rmse; done
# 5. Ω retrained checkpoint (gated repo; you have access) — separate backbone name => separate caches, rows, provenance
huggingface-cli download facebook/VGGT-Omega vggt_omega_1b_512_reproduce.pt --local-dir third_party/checkpoints
sha256sum third_party/checkpoints/vggt_omega_1b_512_reproduce.pt
for s in chess fire heads office pumpkin redkitchen stairs; do CUDA_VISIBLE_DEVICES=1 python experiments/pilot_a.py --gt data/gt/7scenes_${s}_seq01.npz \
   --backbone vggt_omega_repro --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 --rows chained,smr,smr_pgo \
   --json outputs/omega_repro/7scenes/${s}_seq01.json --save-est outputs/omega_repro/7scenes/${s}_seq01.npz; done
python scripts/paired_stats.py rows --reports 'outputs/omega_repro/7scenes/*.json' --row-a chained --row-b smr --metric ate_rmse --per-item
```
Send back: the `grep` of step 2, `rel_7scenes_test.md`, `rel_co3d.md`, the four paired outputs of step 3, the
`ls -la` of the nvs dirs, `gate_sweep.md`, the two `rel_*_useful.md`, the seed comparisons, and the Ω-repro paired output.

## 4. Predictions (scoreboard)
P75 ✓ (original Ω checkpoint). P76 ✗ as stated: strict precision 0.48, not ≥ 0.9; but the budget never fired, so
"most misses are no-proposal" was also wrong (0 no-proposal misses; the misses are 1 no-verified + 1 other).
P79 ✓ for π³ (identity) and VGGT (+0.24 dB, CI excludes 0). P77/P78 pending the sweeps.
P80 with `useful`: ≥ 80 % of accepted seq-01 closures are useful, including all three pumpkin closures the strict
    test calls false. P81 test split (fixed GT): closure precision (covis) 0.6–0.8, 1–3 harmful closures on office /
    redkitchen where +SMR is worse than raw. P82 Ω-repro seq-01: raw ATE within ±0.005 of 0.045; +SMR ordering of scenes
    unchanged (pumpkin, office gain; others ±2 mm). P83 STream3R `gso_reads1.jsonl` holds reads = 1 → Table 3 row reversed.
