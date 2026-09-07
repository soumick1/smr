#!/bin/bash
# Memory-policy ablations over cached backbone passes.  Each variant re-runs pilot_a with one knob changed, on
# 7-Scenes (Table 2 configuration: seq01, stride 5, chunk 32/16, VGGT-Omega) and on CO3D orbits (N=200, VGGT).
# Passes are cached, so a policy variant costs ~10 s per 7-Scenes sequence and ~15 s per orbit; the window-size
# variants (w16, w64) need new passes (~1 min per sequence) and run last.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/ablate_all.sh                 # everything (~2.5 h incl. window sizes)
#   VARIANTS="default sites1 sites3" DATASETS=7scenes bash scripts/ablate_all.sh
#   NSEQ=12  -> CO3D orbits used (default 12 of the 37; ALL=1 for all)
cd "$(dirname "$0")/.."; source .venv/bin/activate; ROOT=${ROOT:-outputs/ablate}; mkdir -p $ROOT outputs/reports
step() { echo "[$(date +%H:%M:%S)] $*"; }
declare -A FLAGS=(
  [default]=""                       [sites1]="--sites 1"                 [sites3]="--sites 3"
  [corr_jump]="--correction jump"    [corr_dist]="--correction distribute" [corr_none]="--correction none"
  [no_robust]="--no-robust-batch"    [remeasure]="--remeasure"            [desc_rgb]="--descriptor rgb"
  [Nh512]="--N-h 512"                [Nh8192]="--N-h 8192"                [torus16]="--torus-N 16"  [torus64]="--torus-N 64"
  [w16]="--chunk 16 --overlap 8"     [w64]="--chunk 64 --overlap 32"
  [desc_dino]="--descriptor dino"    [desc_feat]="--descriptor feat"
)
VARIANTS=${VARIANTS:-"default sites1 sites3 corr_jump corr_dist corr_none no_robust remeasure desc_rgb Nh512 Nh8192 torus16 torus64 w16 w64"}
DATASETS=${DATASETS:-"7scenes co3d"}
SCENES7=${SCENES7:-"chess fire heads office pumpkin redkitchen stairs"}
for v in $VARIANTS; do
  extra=${FLAGS[$v]}; [ -z "${FLAGS[$v]+x}" ] && { step "unknown variant $v"; continue; }
  chunk="--chunk 32 --overlap 16"; case $v in w16|w64) chunk="";; esac
  for ds in $DATASETS; do
    out=$ROOT/$ds/$v; mkdir -p $out
    if [ $ds = 7scenes ]; then
      for s in $SCENES7; do GT=data/gt/7scenes_${s}_seq01.npz; [ -f $GT ] || continue
        [ -f $out/$s.json ] && continue
        step "$v / 7scenes $s"
        python experiments/pilot_a.py --gt $GT --backbone ${SLAMBB:-vggt_omega} --keyframe-stride 5 $chunk --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json > $out/$s.log 2>&1 || { step "  FAILED: $(tail -n 2 $out/$s.log | tr '\n' ' ')"; }
      done
    else
      list=$(ls data/gt/co3d_full/*.npz | { [ -n "${ALL:-}" ] && cat || head -${NSEQ:-12}; })
      for GT in $list; do s=$(basename $GT .npz)
        [ -f $out/$s.json ] && continue
        step "$v / co3d $s"
        python experiments/pilot_a.py --gt $GT --backbone ${BB:-vggt} --keyframe-stride 1 --max-frames 200 $chunk --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json > $out/$s.log 2>&1 || { step "  FAILED: $(tail -n 2 $out/$s.log | tr '\n' ' ')"; }
      done
    fi
  done
done
[ "$ROOT" = outputs/ablate ] && python scripts/ablate_table.py --root outputs/ablate --out paper/tab_ablation.tex || true
