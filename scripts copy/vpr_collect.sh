#!/bin/bash
# After all three screens print DONE: one summary file to send back (tables for both modes, paired deltas vs DINO,
# and the two re-done reliability outputs).
cd ~/smr && source .venv/bin/activate
OUT=outputs/vpr_summary.md; : > $OUT
for m in anchored plain; do
  echo "## VPR sweep, mode = $m" >> $OUT
  python scripts/gate_table.py --root outputs/ablate/vpr/$m --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --md /tmp/vpr_$m.md > /dev/null 2>&1; cat /tmp/vpr_$m.md >> $OUT; echo >> $OUT
done
echo "## Paired vs DINO (gate off), plain mode, 7-Scenes ATE (m) and CO3D AUC@30" >> $OUT
for arm in dino05 eigenplaces cosplace salad@4096 boq@4096 netvlad mixvpr dino+eigenplaces dino+salad@4096; do
  echo "### $arm" >> $OUT
  python scripts/paired_stats.py reports --a 'outputs/ablate/vpr/plain/7scenes/dino/*.json' --b "outputs/ablate/vpr/plain/7scenes/$arm/*.json" --row smr --metric ate_rmse 2>&1 | grep -E "^n=|improved" >> $OUT
  python scripts/paired_stats.py reports --a 'outputs/ablate/vpr/plain/co3d/dino/*.json' --b "outputs/ablate/vpr/plain/co3d/$arm/*.json" --row smr --metric auc30 2>&1 | grep -E "^n=|improved" >> $OUT
done
echo "## Reliability, plain_seed0 / plain_seed1 (re-done with the resolver fix)" >> $OUT
for v in plain_seed0 plain_seed1; do
  echo "### $v" >> $OUT
  python scripts/closure_reliability.py --glob "outputs/ablate/localfrom/7scenes/$v/*.json" --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --est-dir outputs/ablate/localfrom/7scenes/$v 2>&1 | sed -n '/POOLED/,$p' >> $OUT
done
echo "written: $OUT ($(wc -l < $OUT) lines)"
