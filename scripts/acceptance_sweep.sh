#!/bin/bash
# v200: which acceptance rule to adopt. Paper configuration everywhere (address, pairs at +-3, two pairs, Sim(3) interpolation,
# IRLS); only the acceptance test changes. Arms:
#   ag_T_L        agreement test alone: rotation <= T deg, position <= L E_k      (T in 5 10 20 off; L in 0.25 0.5 1.0 off)
#   gates_T_L     pair-validity gates + drift budget (legacy) AND the agreement test at T / L
#   legacy        the pre-v198 rule, for reference
# Scenes are split across GPUs by the caller (SCENES7 / DATASETS), never the same scene on two GPUs at once.
#   CUDA_VISIBLE_DEVICES=0 SCENES7="chess fire heads" DATASETS=7scenes bash scripts/acceptance_sweep.sh &
#   CUDA_VISIBLE_DEVICES=1 SCENES7="office pumpkin redkitchen stairs" DATASETS=7scenes bash scripts/acceptance_sweep.sh &
#   CUDA_VISIBLE_DEVICES=2 DATASETS=co3d bash scripts/acceptance_sweep.sh &
# Then: python scripts/acceptance_pick.py --root outputs/ablate/acceptance
cd "$(dirname "$0")/.."; source .venv/bin/activate; ROOT=${ROOT:-outputs/ablate/acceptance}; mkdir -p $ROOT
step() { echo "[$(date +%H:%M:%S)] $*"; }
declare -A FLAGS; ARMS=""
for T in 5 10 20 1e9; do for L in 0.25 0.5 1.0 1e9; do
  nT=${T/1e9/off}; nL=${L/1e9/off}
  FLAGS[ag_${nT}_${nL}]="--paper --site-agree $T,$L"; ARMS="$ARMS ag_${nT}_${nL}"
done; done
for TL in "5,0.25" "10,0.5" "20,1.0" "1e9,1e9"; do nm=${TL/1e9,1e9/off_off}; nm=${nm/,/_}
  FLAGS[gates_$nm]="--paper --paper-gates --site-agree $TL"; ARMS="$ARMS gates_$nm"
done
FLAGS[legacy]="--legacy"; ARMS="$ARMS legacy"
ARMS=${ARMS_OVERRIDE:-$ARMS}; DATASETS=${DATASETS:-"7scenes co3d"}; SCENES7=${SCENES7:-"chess fire heads office pumpkin redkitchen stairs"}
for v in $ARMS; do extra=${FLAGS[$v]}
  for ds in $DATASETS; do out=$ROOT/$ds/$v; mkdir -p $out
    if [ $ds = 7scenes ]; then
      for s in $SCENES7; do GT=data/gt/7scenes_${s}_seq01.npz; [ -f $GT ] || continue; [ -f $out/$s.json ] && continue
        step "$v / 7scenes $s"
        python experiments/pilot_a.py --gt $GT --backbone ${SLAMBB:-vggt_omega} --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || step "  FAILED: $(grep -m1 -E 'Error|Traceback' $out/$s.log | cut -c1-140)"
      done
    else
      for GT in $(ls data/gt/co3d_full/*.npz | head -${NSEQ:-12}); do s=$(basename $GT .npz); [ -f $out/$s.json ] && continue
        step "$v / co3d $s"
        python experiments/pilot_a.py --gt $GT --backbone ${BB:-vggt} --keyframe-stride 1 --max-frames 200 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || step "  FAILED: $(grep -m1 -E 'Error|Traceback' $out/$s.log | cut -c1-140)"
      done
    fi
  done
done
step "done ($DATASETS: $SCENES7)"
