#!/bin/bash
# v192: distortion-gated window geometry vs anchored vs plain, on cached passes (~30 min on one GPU).
#   CUDA_VISIBLE_DEVICES=0 bash scripts/gated_sweep.sh
# Arms (7-Scenes seq-01 x 7 with VGGT-Omega; CO3D 12 orbits with VGGT):
#   anchored_log            default proposals, anchored geometry (identical to the paper's default; logs distortion)
#   gated05 gated1 gated2 gated5   --local-from gated with --distortion-gate 0.5 / 1 / 2 / 5 deg
#   gated1_seed1            the seed that destroyed chess (0.459 m anchored, 0.017 plain)
#   gated1_cosplace gated1_boq gated1_dinosalad   the descriptors that poisoned windows (0.263 / 0.127 / 0.133 anchored)
# Then:
#   python scripts/gate_table.py --root outputs/ablate/gated --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --md outputs/gated.md
#   for v in gated05 gated1 gated2 gated5; do python scripts/paired_stats.py reports --a 'outputs/ablate/gated/7scenes/anchored_log/*.json' --b "outputs/ablate/gated/7scenes/$v/*.json" --row smr --metric ate_rmse | tail -2; done
#   for v in gated05 gated1 gated2 gated5; do python scripts/paired_stats.py reports --a 'outputs/ablate/gated/co3d/anchored_log/*.json' --b "outputs/ablate/gated/co3d/$v/*.json" --row smr --metric auc30 | tail -2; done
#   for ds in 7scenes co3d; do python scripts/closure_reliability.py --glob "outputs/ablate/gated/$ds/anchored_log/*.json" --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --est-dir outputs/ablate/gated/$ds/anchored_log | grep -E "anchored_distortion|POOLED"; done
cd "$(dirname "$0")/.."; source .venv/bin/activate; ROOT=${ROOT:-outputs/ablate/gated}; mkdir -p $ROOT
python experiments/pilot_a.py --help 2>/dev/null | grep -q -- "--distortion-gate" || { echo "pilot_a lacks --distortion-gate: run  python scripts/apply_all_patches.py  first"; exit 2; }
step() { echo "[$(date +%H:%M:%S)] $*"; }
declare -A FLAGS=(
  [anchored_log]="--local-from anchored --index template --seed 0"
  [gated05]="--local-from gated --distortion-gate 0.5 --index template --seed 0"
  [gated1]="--local-from gated --distortion-gate 1.0 --index template --seed 0"
  [gated2]="--local-from gated --distortion-gate 2.0 --index template --seed 0"
  [gated5]="--local-from gated --distortion-gate 5.0 --index template --seed 0"
  [gated1_seed1]="--local-from gated --distortion-gate 1.0 --index template --seed 1"
  [gated1_cosplace]="--local-from gated --distortion-gate 1.0 --index template --seed 0 --descriptor cosplace --desc-thresh -1"
  [gated1_boq]="--local-from gated --distortion-gate 1.0 --index template --seed 0 --descriptor boq@4096 --desc-thresh -1"
  [gated1_dinosalad]="--local-from gated --distortion-gate 1.0 --index template --seed 0 --descriptor dino+salad@4096 --desc-thresh -1"
)
ARMS=${ARMS:-"anchored_log gated05 gated1 gated2 gated5 gated1_seed1 gated1_cosplace gated1_boq gated1_dinosalad"}
DATASETS=${DATASETS:-"7scenes co3d"}; SCENES7=${SCENES7:-"chess fire heads office pumpkin redkitchen stairs"}
for v in $ARMS; do extra=${FLAGS[$v]}; [ -z "${FLAGS[$v]+x}" ] && { step "unknown arm $v"; continue; }
  for ds in $DATASETS; do out=$ROOT/$ds/$v; mkdir -p $out
    if [ $ds = 7scenes ]; then
      for s in $SCENES7; do GT=data/gt/7scenes_${s}_seq01.npz; [ -f $GT ] || continue; [ -f $out/$s.json ] && [ -f $out/$s.npz ] && continue
        step "$v / 7scenes $s"
        python experiments/pilot_a.py --gt $GT --backbone ${SLAMBB:-vggt_omega} --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || step "  FAILED: $(grep -m1 -E 'Error|needs' $out/$s.log | cut -c1-140)"
      done
    else
      list=$(ls data/gt/co3d_full/*.npz | { [ -n "${ALL:-}" ] && cat || head -${NSEQ:-12}; })
      for GT in $list; do s=$(basename $GT .npz); [ -f $out/$s.json ] && [ -f $out/$s.npz ] && continue
        step "$v / co3d $s"
        python experiments/pilot_a.py --gt $GT --backbone ${BB:-vggt} --keyframe-stride 1 --max-frames 200 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || step "  FAILED: $(grep -m1 -E 'Error|needs' $out/$s.log | cut -c1-140)"
      done
    fi
  done
done
step "done"
