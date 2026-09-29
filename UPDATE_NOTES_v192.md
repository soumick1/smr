# smr_updates192 — distortion-gated window geometry (reviewer question: why wrong proposals corrupt the window)

* `scripts/apply_v192_patch.py` (run by `apply_all_patches.py`): every window with anchor frames now logs
  `anchored_distortion` {rot_deg, pos_rel, used} in all modes (the Sim(3) residual between the window's frames in the
  joint pass and in its own cached pass); new mode `--local-from gated --distortion-gate DEG` keeps the joint-pass
  geometry only when that residual is below the gate (default 1 deg; correct anchors measured 0.11 deg median, 0.31 max;
  the chess seed-1 poisoning was 46 deg). Verified: gate 1e9 reproduces anchored, gate 0 reproduces plain; 11 tests pass.
* `scripts/gated_sweep.sh`: anchored (with logging), gated 0.5 / 1 / 2 / 5 deg, gated 1 deg with seed 1 and with the
  three poisoning descriptors, both datasets, cached passes (~30 min).
* `scripts/closure_reliability.py`: reports distortion median / p90 / max and the number of windows reverted to plain.
* `QA_anchored_pass_distortion_2026-09-18.md`: the answer (mechanism, evidence, strategies, draft text); gated numbers pending.

```bash
unzip -o smr_updates192.zip && python scripts/apply_all_patches.py          # six PASS lines
PYTHONPATH=src python -m pytest -q tests/test_guarantees.py
CUDA_VISIBLE_DEVICES=0 bash scripts/gated_sweep.sh
python scripts/gate_table.py --root outputs/ablate/gated --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --md outputs/gated.md
for v in gated05 gated1 gated2 gated5; do echo == $v; python scripts/paired_stats.py reports --a 'outputs/ablate/gated/7scenes/anchored_log/*.json' --b "outputs/ablate/gated/7scenes/$v/*.json" --row smr --metric ate_rmse | tail -2; python scripts/paired_stats.py reports --a 'outputs/ablate/gated/co3d/anchored_log/*.json' --b "outputs/ablate/gated/co3d/$v/*.json" --row smr --metric auc30 | tail -2; done
for ds in 7scenes co3d; do echo == $ds; python scripts/closure_reliability.py --glob "outputs/ablate/gated/$ds/anchored_log/*.json" --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --est-dir outputs/ablate/gated/$ds/anchored_log | grep -E "anchored_distortion|reverted"; done
```
Send back `outputs/gated.md` and the two loops' output.
