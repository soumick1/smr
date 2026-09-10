# smr_updates179 — patches that land, resolver and usefulness fixes, and what the second run settled

## 1. Why the sweep and the Ω rerun failed again, and the fix
Neither patch had landed: `pilot_a` still lacked the flags and `vggt_omega_repro` was not registered, while the v178
`closure_reliability.py` clearly ran (seq03 -> seq03 GT). The v177 patch used exact v167 text as anchors and refuses
when your v176 `pilot_a.py` differs at one of them; the `&&` chain then skipped the v178 patch. Both are rewritten:
`apply_v177_patch.py` now finds the argparse block and the `common = dict(...)` / `report = dict(...)` calls by regex and
parenthesis matching (tested on v167 and on a simulated v170-style block with extra kwargs and a trailing comma), and
`apply_v178_patch.py` never refuses (falls back to appending the import). If v177 still refuses it prints the reason and
the lines around it; paste that. Also: `hf download` replaces the deprecated `huggingface-cli`.

Other fixes: `gate_table.py` crashed on the new `find_gt` signature; the GT resolver preferred `7scenes_<scene>_all.npz`
over `_seq01.npz` on a tie (multi-session files are now excluded unless the report names them, candidates are validated by
rebuilding the report's windows, and `--gt-pattern '7scenes_{stem}_seq01.npz'` pins it explicitly); the "useful" metric
now measures the placement error of the two closed windows (mean over frame pairs of the relative-position error after
each trajectory's own Sim(3) to GT, metres; useful/harmful at ±10 %) instead of the angular pair error, which is
dominated by within-window noise; `paired_stats --macro` reproduces the handover's per-scene-then-over-scenes average.

## 2. What the second run settled
* **Mutual agreement test**: no `"disagree"` in any Table 2 / test-split / CO3D report -> it was off everywhere (code and paper now agree: App. D Eqs. 5–6 are the off-by-default arm).
* **Test split (18 trajectories, correct GT)**: 217 sites, 124 verified (extent gate 90, direction 9); 80 closures tested, 77 accepted (3 rot rejections). Closure precision: strict 0.34, 2x 0.58, **covis 0.83** (13 false by co-visibility); recall strict 1.00, covis 0.75. Raw -> +SMR: **0.0547 -> 0.0500** (Δ −0.0047 [−0.0171, +0.0056], 9/18 improved, median +0.0004, sign p = 1); +PGO 0.0447 (Δ −0.0099 [−0.0232, −0.0000]). In the handover's macro convention the same reports give 0.049 -> 0.042 -> 0.038; the paper must state which average it reports (`--macro` prints both). Losers: office_seq02/06/09 (+0.014/+0.012/+0.022), redkitchen_seq14 (+0.035, whose closures are all "true" by covis, i.e. valid revisits measured or applied badly), redkitchen_seq03 (+0.006). Winners: pumpkin_seq01 (−0.084), office_seq07 (−0.039), redkitchen_seq12 (−0.019), fire_seq03/04, chess_seq03/05.
* **Scaffold vs flat, test split**: 0.0500 vs 0.0469 (flat better, Δ −0.0031 [−0.0097, +0.0020], 9/18 vs 8/18). With seq-01 (Δ +0.0027) and CO3D (Δ −0.07 AUC) this is three ties; Table 5 becomes a tie in the text as well as the numbers.
* **CO3D (12 orbits, correct GT)**: closure precision strict 0.67, 2x 0.94, covis 0.94 (1 false); recall strict 0.82; extent gate rejects 118/140 sites; median span 7 windows.
* **Usefulness, seq-01 (6 scenes; office's `_all` GT failed, fixed now)**: with the angular metric only 7/22 closures moved the pair error by > 0.5°, 4 worsened it; pumpkin 1/3 useful despite ATE 0.160 -> 0.077. That metric is the wrong lens (noise-dominated); the placement-error version (v179) is what to quote. Rerun (step 3).
* **Seeds (Task 3), 7-Scenes seq-01, flat 0.0332 vs scaffold seed 0/1/2 = 0.0305 / 0.0924 / 0.0302.** Seed 1 blows up: mean loss when worsened 0.111, i.e. one or two scenes went to ~0.3–0.4 m. **The scaffold's random projection is not innocuous** (P78 wrong). Find the scene (step 3: `--per-item`), diff its closures against seed 0 (`compare_closures.py`), and check whether the culprit is a false closure the flat index does not make. CO3D seeds are not compared yet (step 3).
* **NVS**: the `reads` field inside `stream3r/gso_reads1.jsonl` is 1 and `gso_reads4.jsonl` is 4, files written 02:50 and 11:22 on Sep 7 -> **Table 3's STream3R row is swapped** (raw 18.79, read 18.60; P83 ✓). The SSIM/LPIPS lines were cut by `| head -4` in the command I gave; rerun (step 3). π³ identity stands.
* **Ω**: still the original checkpoint on disk; the retrained one is not downloaded yet (step 4).

## 3. Commands, in order
```bash
unzip -o smr_updates179.zip
python scripts/apply_v177_patch.py --check && python scripts/apply_v177_patch.py     # if it REFUSES, paste its output
python scripts/apply_v178_patch.py
PYTHONPATH=src python -m pytest -q tests/test_guarantees.py
python -c "import sys; sys.path.insert(0,'src'); import smr.backbones as b; print('vggt_omega_repro' in b.base._REGISTRY)"   # True
python experiments/pilot_a.py --help | grep -c "budget-rot"                                                                # 1
# seeds: which scene blew up under seed 1, and why
for s in 0 1 2; do python scripts/paired_stats.py reports --a 'outputs/ablate/seeds/7scenes/flat/*.json' --b "outputs/ablate/seeds/7scenes/seed$s/*.json" --row smr --metric ate_rmse --per-item | tail -9; done
python scripts/paired_stats.py reports --a 'outputs/ablate/seeds/co3d/flat/*.json' --b 'outputs/ablate/seeds/co3d/seed1/*.json' --row smr --metric auc30 --per-item
SC=<worst scene from the seed1 list>; python scripts/compare_closures.py outputs/ablate/seeds/7scenes/seed0/$SC.json outputs/ablate/seeds/7scenes/seed1/$SC.json
python scripts/closure_reliability.py --glob 'outputs/ablate/seeds/7scenes/seed1/*.json' --gt-dir data/gt --gt-pattern '7scenes_{stem}_seq01.npz' --est-dir outputs/ablate/seeds/7scenes/seed1 --md outputs/rel_seed1.md
# NVS STream3R, all three metrics (no head)
for m in psnr ssim lpips; do python scripts/paired_stats.py nvs --a outputs/nvs/stream3r/gso_reads1.jsonl --b outputs/nvs/stream3r/gso_reads4.jsonl --metric $m | tail -2; done
# usefulness with the placement metric (seq-01 default runs of the gate sweep already have npz)
python scripts/closure_reliability.py --glob 'outputs/ablate/gate/7scenes/default/*.json' --gt-dir data/gt --gt-pattern '7scenes_{stem}_seq01.npz' --est-dir outputs/ablate/gate/7scenes/default --md outputs/rel_default_useful.md
# test split with usefulness needs npz: rerun the 18 template reports with --save-est (cached passes, ~10 s each) into a new dir
mkdir -p outputs/7scenes_test_est; for f in outputs/7scenes_test/*_vggt_omega_template.json; do b=$(basename $f _vggt_omega_template.json); \
  GT=$(ls data/gt/7scenes_${b}.npz); python experiments/pilot_a.py --gt $GT --backbone vggt_omega --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 \
  --rows chained,smr,smr_pgo --json outputs/7scenes_test_est/$b.json --save-est outputs/7scenes_test_est/$b.npz > /dev/null 2>&1 || echo FAILED $b; done
python scripts/closure_reliability.py --glob 'outputs/7scenes_test_est/*.json' --gt-dir data/gt --est-dir outputs/7scenes_test_est --md outputs/rel_test_useful.md
python scripts/paired_stats.py rows --reports 'outputs/7scenes_test/*_vggt_omega_template.json' --row-a chained --row-b smr --metric ate_rmse --macro
# gate sweep (now the flags exist) and table
CUDA_VISIBLE_DEVICES=0 bash scripts/gate_sweep.sh && python scripts/gate_table.py --root outputs/ablate/gate --gt-dir data/gt --gt-pattern '7scenes_{stem}_seq01.npz' --md outputs/gate_sweep.md
# 4. Ω retrained checkpoint
hf download facebook/VGGT-Omega vggt_omega_1b_512_reproduce.pt --local-dir third_party/checkpoints && sha256sum third_party/checkpoints/vggt_omega_1b_512_reproduce.pt
for s in chess fire heads office pumpkin redkitchen stairs; do CUDA_VISIBLE_DEVICES=1 python experiments/pilot_a.py --gt data/gt/7scenes_${s}_seq01.npz \
  --backbone vggt_omega_repro --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 --rows chained,smr,smr_pgo \
  --json outputs/omega_repro/7scenes/${s}_seq01.json --save-est outputs/omega_repro/7scenes/${s}_seq01.npz; done
python scripts/paired_stats.py rows --reports 'outputs/omega_repro/7scenes/*.json' --row-a chained --row-b smr --metric ate_rmse --per-item
```
Note the test-split GT file for the rerun loop: if `data/gt/7scenes_<scene>_seqNN.npz` is not the naming, adjust `GT=`;
the reliability output prints which file it used.

## 4. Scoreboard
P78 ✗ (seed 1 mean 0.092; seeds are not interchangeable). P83 ✓ (STream3R row swapped). P80/P81/P82 pending.
P84 the seed-1 blow-up is a single scene with one false closure accepted at a long span (n_stretch ≥ 4) that the flat
    index does not propose; removing that one closure recovers the seed-0 number. P85 test-split usefulness: ≥ 60 % of
    accepted closures reduce the window-pair placement error; the five losing sequences each carry ≥ 1 harmful closure.
