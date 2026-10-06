#!/bin/bash
# v198: what the manuscript's configuration costs or gains against the legacy one, on the controlled blocks.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/paper_config_sweep.sh            # ~1 h: new address -> new proposals -> some new anchored passes
#   ARMS="legacy paper" bash scripts/paper_config_sweep.sh                # subset
# Arms:
#   legacy        pre-v198 configuration (must reproduce 7-Scenes 0.030 / 0.029 and CO3D 93.4 / 93.3)
#   paper         the manuscript: 1024/64 address over 7,824-D state (orientation rings), median-depth units, pairs at +-3,
#                 two pairs required, agreement 3 deg / 0.15 E as the only acceptance test, Sim(3) interpolation, IRLS fit + scale gate
#   paper_gates   paper + the legacy per-pair validity gates and drift budget kept
#   paper_posonly paper with position-only scaffold state (ring-N 0)
#   paper_relax   paper with the local pose-graph relaxation instead of Sim(3) interpolation
#   paper_oldaddr paper rules with the legacy address (2048 / 128 / torus 32, no rings)
# Then:
#   python scripts/gate_table.py --root outputs/ablate/paper_config --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --md outputs/paper_config.md
#   for v in paper paper_gates paper_posonly paper_relax paper_oldaddr; do echo == $v; python scripts/paired_stats.py reports --a 'outputs/ablate/paper_config/7scenes/legacy/*.json' --b "outputs/ablate/paper_config/7scenes/$v/*.json" --row smr --metric ate_rmse | tail -2; python scripts/paired_stats.py reports --a 'outputs/ablate/paper_config/co3d/legacy/*.json' --b "outputs/ablate/paper_config/co3d/$v/*.json" --row smr --metric auc30 | tail -2; done
cd "$(dirname "$0")/.."; source .venv/bin/activate; ROOT=${ROOT:-outputs/ablate/paper_config}; mkdir -p $ROOT
python experiments/pilot_a.py --help 2>/dev/null | grep -q -- "--legacy" || { echo "pilot_a lacks --paper/--legacy: unpack smr_updates198.zip first"; exit 2; }
step() { echo "[$(date +%H:%M:%S)] $*"; }
declare -A FLAGS=(
  [legacy]="--legacy"
  [paper]="--paper"
  [paper_gates]="--paper --paper-gates"
  [paper_posonly]="--paper --ring-N 0"
  [paper_relax]="--paper --correction relax"
  [paper_oldaddr]="--paper --N-h 2048 --torus-N 32 --k 128 --ring-N 0"
)
ARMS=${ARMS:-"legacy paper paper_gates paper_posonly paper_relax paper_oldaddr"}; DATASETS=${DATASETS:-"7scenes co3d"}
SCENES7=${SCENES7:-"chess fire heads office pumpkin redkitchen stairs"}
for v in $ARMS; do extra=${FLAGS[$v]}; [ -z "${FLAGS[$v]+x}" ] && { step "unknown arm $v"; continue; }
  for ds in $DATASETS; do out=$ROOT/$ds/$v; mkdir -p $out
    if [ $ds = 7scenes ]; then
      for s in $SCENES7; do GT=data/gt/7scenes_${s}_seq01.npz; [ -f $GT ] || continue; [ -f $out/$s.json ] && continue
        step "$v / 7scenes $s"
        python experiments/pilot_a.py --gt $GT --backbone ${SLAMBB:-vggt_omega} --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || step "  FAILED: $(grep -m1 -E 'Error|Traceback' $out/$s.log | cut -c1-140)"
      done
    else
      list=$(ls data/gt/co3d_full/*.npz | { [ -n "${ALL:-}" ] && cat || head -${NSEQ:-12}; })
      for GT in $list; do s=$(basename $GT .npz); [ -f $out/$s.json ] && continue
        step "$v / co3d $s"
        python experiments/pilot_a.py --gt $GT --backbone ${BB:-vggt} --keyframe-stride 1 --max-frames 200 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || step "  FAILED: $(grep -m1 -E 'Error|Traceback' $out/$s.log | cut -c1-140)"
      done
    fi
  done
done
step "done"
