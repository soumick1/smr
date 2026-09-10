# smr_updates181 — one command applies everything; local_from without a constructor change

## What went wrong in the last run
`apply_v180_patch.py` was not applied (tests: `unexpected keyword argument 'local_from'`; sweep: `unrecognized
arguments: --local-from`). Whether it was skipped or refused I cannot tell from the paste, so v181 removes both
possibilities: one script applies every patch and prints a status line per patch plus PASS/FAIL checks, the stitcher
change no longer touches the constructor (a class attribute `AnchoredStitcher.local_from`, set by `pilot_a` from the flag),
the two local_from tests SKIP with a message when the tree is unpatched instead of failing, and the sweep refuses to
start against an unpatched tree. Verified here on v167 and on a simulated v173-style file (extra constructor kwargs,
candidates in the event dict, extra `cache.get` argument). Through `pilot_a`: `--local-from plain --correction none`
reproduces the chained row exactly; `--local-from plain` closes the same loops with a different trajectory; every
window with anchors logs `anchored_distortion`; provenance records the flag.

## Seed-1 autopsy (from your `compare_closures` output)
Only 4 of 12 windows had the same candidates as seed 0; both seed-1 closures were rejected; the trajectory still went
0.025 -> 0.459 (AUC30 95.6 -> 45.1). Wrong proposals poisoned the windows through the enlarged passes; no closure was
involved. This is the case `--local-from plain` is designed to remove.

## Run (three commands, in this order; paste all output)
```bash
unzip -o smr_updates181.zip
python scripts/apply_all_patches.py            # must end with four PASS lines
PYTHONPATH=src python -m pytest -q tests/test_guarantees.py   # 11 passed (2 would say SKIPPED if the patch were missing)
```
Then the sweep and comparisons (cached passes, ~25 min):
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/local_from_sweep.sh
python scripts/paired_stats.py rows --reports 'outputs/ablate/localfrom/7scenes/plain_none/*.json' --row-a chained --row-b smr --metric ate_rmse            # delta 0.0000 expected
python scripts/paired_stats.py rows --reports 'outputs/ablate/localfrom/7scenes/anch_none/*.json' --row-a chained --row-b smr --metric ate_rmse --per-item  # anchored-pass effect alone
for v in plain_seed0 plain_seed1 plain_seed2 plain_flat; do echo == $v; python scripts/paired_stats.py reports --a 'outputs/ablate/seeds/7scenes/flat/*.json' --b "outputs/ablate/localfrom/7scenes/$v/*.json" --row smr --metric ate_rmse --per-item | tail -9; done
python scripts/paired_stats.py reports --a 'outputs/ablate/gate/7scenes/default/*.json' --b 'outputs/ablate/localfrom/7scenes/plain_seed0/*.json' --row smr --metric ate_rmse --per-item
python scripts/paired_stats.py reports --a 'outputs/ablate/gate/co3d/default/*.json' --b 'outputs/ablate/localfrom/co3d/plain_seed0/*.json' --row smr --metric auc30 --per-item
for v in plain_seed0 plain_seed1; do python scripts/closure_reliability.py --glob "outputs/ablate/localfrom/7scenes/$v/*.json" --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --est-dir outputs/ablate/localfrom/7scenes/$v --md outputs/rel_$v.md | tail -45; done
python scripts/closure_reliability.py --glob 'outputs/7scenes_test_est/*.json' --gt-dir data/gt --est-dir outputs/7scenes_test_est | tail -30      # harmful-closure breakdown
for m in psnr ssim lpips; do python scripts/paired_stats.py nvs --a outputs/nvs/stream3r/val_reads1.jsonl --b outputs/nvs/stream3r/val_reads4.jsonl --metric $m | tail -2; done
hf download facebook/VGGT-Omega --include "vggt_omega_1b_512_reproduce*" --local-dir third_party/checkpoints && ls -la third_party/checkpoints/ | grep reproduce
```
Predictions P86–P88 stand as in v180.
