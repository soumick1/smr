# smr_updates180 — the anchored-pass problem, a fix (`--local-from plain`), and what the third run settled

## 1. The finding that changes the method description (and probably Table 2's tail)
Seed 1 on chess: **0 closures accepted** (both rejected by the budget), yet ATE 0.025 -> 0.459 and AUC_within 93.0 -> 83.1.
Nothing was written back; the trajectory was destroyed by the anchored passes themselves. A window with proposed sites
takes its own geometry from the enlarged pass (32 window frames + up to 4 anchor frames); when the proposals are wrong
places, VGGT-Ω's joint reconstruction of window + unrelated views distorts the window, the chain inherits it, and no
gate can undo it because no closure was ever applied. This is test B's mechanism showing up in the wild, and it is why
the scaffold seed matters: the seed changes which sites are proposed, not which closures are accepted (CO3D seed 1:
apple_110 −3.8 AUC, bowl_70 −1.8, bowl_69 +2.3; 6/6). P84 was wrong (I predicted one false closure at a long span).

**Fix (v180, `--local-from plain`)**: the window's geometry comes from its own 32-frame pass (already cached from the
raw chain, so no new backbone work); the enlarged pass is used only to *measure* the revisit: its window frames are
fitted to the plain pass by one Sim(3) and the anchor frames are carried through that similarity, so verification,
S_A, S_B, edges, budget and write-back run unchanged in the plain pass's coordinates. Consequences, all tested:
rejected proposals leave the geometry untouched; `--correction none` reproduces the raw chain bit for bit (the genuine
no-write-back control the plan asked for; the junction blend is skipped when there is no correction to smooth); loops
still close (synthetic: chain 0.369, plain 0.137 with 4 loops, anchored 0.151 with 2). Every window with anchors now
logs `anchored_distortion` (rotation / position by which the enlarged pass moved the window's own frames), which is
the diagnostic for seed sensitivity. Default stays `anchored` so every existing number is reproduced; the sweep below
decides whether `plain` becomes the paper's default.

## 2. What the third run settled
* **Gate sweep**: all eight budget cells (rotation and position ×0.5, ×1, ×2) are identical to default on both datasets
  (7-Scenes 0.030/0.029, 4.1 loops, 16/29 useful; CO3D 93.4/93.1, 1.5 loops). The drift budget never binds, even at
  half size (median accepted correction 0.57°, 0.07 spreads vs a floor of 5°/0.5). The **extent gate is the operative
  acceptance test**: 1.5 -> 7-Scenes 0.027/0.027 with 2.0 loops (precision covis 1.00, recall 0.34) but CO3D 91.8 with
  0.1 loops/seq; 6 -> 7-Scenes 0.033/0.031, 5.6 loops, 9 false, 16 harmful; CO3D 93.4/93.6, 4.2 loops, 21 false.
  Junction blend: 7-Scenes 0.032 vs 0.030 online (2 mm), PGO identical; CO3D 93.3 vs 93.4. Top-10 proposals: PGO
  0.026 vs 0.029 with the same closures. Site rotation/direction gates never fire on 7-Scenes. For the paper: Table 6
  gets extent 1.5 / 3 / 6, no-blend and top-10 rows; App. D says the budget is inert on these datasets and the extent
  test plus the appearance-site requirement is the gate.
* **Usefulness (placement error of the two closed windows, metres)**: seq-01 16/29 useful, 6 harmful, error
  0.064 -> 0.043; office 7/7 (0.090 -> 0.037), pumpkin 2/3 (0.234 -> 0.153), redkitchen 1/5, heads 3/7 (2 harmful),
  fire 2/5 (2 harmful). Test split 44/77 useful, 21 harmful, 0.087 -> 0.070; the six losing sequences all carry harmful
  closures (office_seq02 0/5 with 4 harmful; office_seq09 2/6 with 4; redkitchen_seq14 1/3 with 2). **All 6 harmful
  seq-01 closures and 15 of 21 on the split are true revisits by co-visibility**: recognition is not the problem;
  measurement and application of correctly recognised revisits is. `closure_reliability.py` now prints a breakdown of
  useful/harmful by sites (1/2), scale (measured/fixed), span, correction size, cue similarity and prior error.
  Retrieval: recall@1 0.83 / recall@5 0.89 (seq-01), 0.68 / 0.74 (split).
* **STream3R, GSO, 1,033 objects, `reads` fields confirmed**: PSNR 18.79 -> 18.60, SSIM 0.784 -> 0.781, LPIPS
  0.209 -> 0.222, all with CIs excluding zero. Table 3's row is the swap of raw and read on all three metrics.
* **Split aggregation**: macro (per-scene, then over scenes) reproduces the handover's 0.049 -> 0.042; micro over 18
  trajectories is 0.055 -> 0.050. The paper must say which.
* **Ω**: the Hub file is `vggt_omega_1b_512_reproduce_noconf.pt` (4.58 GB, added two days ago), not the name in the
  reproduction notes; `hf download` with `--include "vggt_omega_1b_512_reproduce*"` fetches it. The backbone now
  accepts any `*reproduce*.pt`, loads non-strictly with a printed key list only if strict loading fails, and keeps all
  points if `depth_conf` is absent (a "noconf" checkpoint may have no confidence head).

## 3. Commands
```bash
unzip -o smr_updates180.zip && python scripts/apply_v180_patch.py --check && python scripts/apply_v180_patch.py
PYTHONPATH=src python -m pytest -q tests/test_guarantees.py                                              # 11 pass
python scripts/compare_closures.py outputs/ablate/seeds/7scenes/seed0/chess.json outputs/ablate/seeds/7scenes/seed1/chess.json   # what seed 1 proposed
# local-from sweep: cached passes, ~25 min
CUDA_VISIBLE_DEVICES=0 bash scripts/local_from_sweep.sh
python scripts/paired_stats.py rows --reports 'outputs/ablate/localfrom/7scenes/plain_none/*.json' --row-a chained --row-b smr --metric ate_rmse   # delta 0.0000 expected
python scripts/paired_stats.py rows --reports 'outputs/ablate/localfrom/7scenes/anch_none/*.json' --row-a chained --row-b smr --metric ate_rmse --per-item   # anchored-pass effect alone
for v in plain_seed0 plain_seed1 plain_seed2 plain_flat; do echo == $v; python scripts/paired_stats.py reports --a 'outputs/ablate/seeds/7scenes/flat/*.json' --b "outputs/ablate/localfrom/7scenes/$v/*.json" --row smr --metric ate_rmse --per-item | tail -9; done
python scripts/paired_stats.py reports --a 'outputs/ablate/gate/7scenes/default/*.json' --b 'outputs/ablate/localfrom/7scenes/plain_seed0/*.json' --row smr --metric ate_rmse --per-item    # anchored vs plain, same seed
python scripts/paired_stats.py reports --a 'outputs/ablate/gate/co3d/default/*.json' --b 'outputs/ablate/localfrom/co3d/plain_seed0/*.json' --row smr --metric auc30 --per-item
for v in plain_seed0 plain_seed1; do python scripts/closure_reliability.py --glob "outputs/ablate/localfrom/7scenes/$v/*.json" --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --est-dir outputs/ablate/localfrom/7scenes/$v --md outputs/rel_$v.md | tail -40; done
# harmful-closure breakdown on the existing runs (no GPU)
python scripts/closure_reliability.py --glob 'outputs/7scenes_test_est/*.json' --gt-dir data/gt --est-dir outputs/7scenes_test_est | tail -30
python scripts/closure_reliability.py --glob 'outputs/ablate/gate/7scenes/default/*.json' --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --est-dir outputs/ablate/gate/7scenes/default | tail -30
# STream3R held-out block of Table 3
for m in psnr ssim lpips; do python scripts/paired_stats.py nvs --a outputs/nvs/stream3r/val_reads1.jsonl --b outputs/nvs/stream3r/val_reads4.jsonl --metric $m | tail -2; done
# Ω retrained checkpoint
hf download facebook/VGGT-Omega --include "vggt_omega_1b_512_reproduce*" --local-dir third_party/checkpoints && ls -la third_party/checkpoints/ | grep reproduce
for s in chess fire heads office pumpkin redkitchen stairs; do CUDA_VISIBLE_DEVICES=1 python experiments/pilot_a.py --gt data/gt/7scenes_${s}_seq01.npz \
  --backbone vggt_omega_repro --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 --rows chained,smr,smr_pgo \
  --json outputs/omega_repro/7scenes/${s}_seq01.json --save-est outputs/omega_repro/7scenes/${s}_seq01.npz 2>&1 | grep -E "^chained |^smr |repro\]"; done
python scripts/paired_stats.py rows --reports 'outputs/omega_repro/7scenes/*.json' --row-a chained --row-b smr --metric ate_rmse --per-item
```
If the Ω download reports "file not found" again, run `hf download facebook/VGGT-Omega --include "*.pt" --local-dir /tmp/omega_ls --dry-run 2>/dev/null || python -c "from huggingface_hub import list_repo_files; print(list_repo_files('facebook/VGGT-Omega'))"` and send the listing.

## 4. Scoreboard
P84 ✗ (zero closures, poisoned passes). P85 ½ (57 % useful; all losers carry harmful closures ✓). P83 ✓ (all three metrics).
P86 with `plain`, seed-1 chess returns to ≤ 0.03 and seeds 0/1/2 agree within ±0.003 m.
P87 `anch_none` differs from raw by > 2 mm on ≥ 2 scenes (the anchored-pass effect alone); `plain_seed0` is within
    ±0.002 of anchored default on seq-01 and reduces the split's harmful count by ≥ 1/3.
P88 `anchored_distortion` median rotation ≤ 1° on 7-Scenes default windows; > 5° on seed-1 chess windows.
