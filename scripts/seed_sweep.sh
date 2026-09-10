#!/bin/bash
# v177 (plan Task 3): the scaffold index under three random projections (--seed 0 1 2) vs the flat index,
# same cached passes as scripts/ablate_all.sh.  ~15 min.  Then:
#   for s in 0 1 2; do python scripts/paired_stats.py reports --a "outputs/ablate/seeds/7scenes/flat/*.json" \
#       --b "outputs/ablate/seeds/7scenes/seed$s/*.json" --row smr --metric ate_rmse --per-item; done
#   python scripts/compare_closures.py outputs/ablate/seeds/7scenes/seed0/pumpkin.json outputs/ablate/seeds/7scenes/flat/pumpkin.json
cd "$(dirname "$0")/.."; source .venv/bin/activate; ROOT=${ROOT:-outputs/ablate/seeds}; mkdir -p $ROOT
step() { echo "[$(date +%H:%M:%S)] $*"; }
declare -A FLAGS=( [flat]="--index flat" [seed0]="--index template --seed 0" [seed1]="--index template --seed 1" [seed2]="--index template --seed 2" )
VARIANTS=${VARIANTS:-"flat seed0 seed1 seed2"}; DATASETS=${DATASETS:-"7scenes co3d"}
SCENES7=${SCENES7:-"chess fire heads office pumpkin redkitchen stairs"}
for v in $VARIANTS; do extra=${FLAGS[$v]}
  for ds in $DATASETS; do out=$ROOT/$ds/$v; mkdir -p $out
    if [ $ds = 7scenes ]; then
      for s in $SCENES7; do GT=data/gt/7scenes_${s}_seq01.npz; [ -f $GT ] || continue; [ -f $out/$s.json ] && continue
        step "$v / 7scenes $s"
        python experiments/pilot_a.py --gt $GT --backbone ${SLAMBB:-vggt_omega} --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || step "  FAILED: $(tail -n 2 $out/$s.log | tr '\n' ' ')"
      done
    else
      list=$(ls data/gt/co3d_full/*.npz | { [ -n "${ALL:-}" ] && cat || head -${NSEQ:-12}; })
      for GT in $list; do s=$(basename $GT .npz); [ -f $out/$s.json ] && continue
        step "$v / co3d $s"
        python experiments/pilot_a.py --gt $GT --backbone ${BB:-vggt} --keyframe-stride 1 --max-frames 200 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || step "  FAILED: $(tail -n 2 $out/$s.log | tr '\n' ' ')"
      done
    fi
  done
done
step "done"
