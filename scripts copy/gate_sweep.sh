#!/bin/bash
# v177 (plan Task 5): sensitivity of the OPERATIVE gate constants on cached backbone passes.
# Same sequences and passes as scripts/ablate_all.sh (7-Scenes seq-01 x 7 with VGGT-Omega, stride 5, W=32/O=16;
# CO3D 12 orbits with VGGT, N=200).  The variants change only acceptance, so every pass is already cached:
# ~10-15 s per sequence, ~1.5 h in total on one GPU, no new backbone passes.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/gate_sweep.sh
#   VARIANTS="default b_r05_p1 b_r2_p1" DATASETS=7scenes bash scripts/gate_sweep.sh
# Then: python scripts/gate_table.py --root outputs/ablate/gate --gt-dir data/gt --md outputs/gate_sweep.md
# Saves --save-est npz next to each report so closure_reliability.py --est-dir can score closure usefulness.
# Requires the v177 pilot_a flags (python scripts/apply_v177_patch.py).
cd "$(dirname "$0")/.."; source .venv/bin/activate; ROOT=${ROOT:-outputs/ablate/gate}; mkdir -p $ROOT
step() { echo "[$(date +%H:%M:%S)] $*"; }
# drift budget grid: rotation x {0.5, 1, 2} and position x {0.5, 1, 2} of the default (10,3,45) / (1.0,0.5)
R05="--budget-rot 5,1.5,22.5"; R2="--budget-rot 20,6,90"; P05="--budget-pos 0.5,0.25"; P2="--budget-pos 2.0,1.0"
declare -A FLAGS=(
  [default]=""
  [b_r05_p05]="$R05 $P05"  [b_r05_p1]="$R05"  [b_r05_p2]="$R05 $P2"
  [b_r1_p05]="$P05"                            [b_r1_p2]="$P2"
  [b_r2_p05]="$R2 $P05"    [b_r2_p1]="$R2"    [b_r2_p2]="$R2 $P2"
  [site_rot05]="--site-rot 5"     [site_rot2]="--site-rot 20"
  [site_dir05]="--site-dir 12.5"  [site_dir2]="--site-dir 50"
  [extent05]="--extent-factor 1.5" [extent2]="--extent-factor 6"
  [nosmooth]="--no-smooth-junctions"
  [top10]="--top-proposals 10"
)
VARIANTS=${VARIANTS:-"default b_r05_p05 b_r05_p1 b_r05_p2 b_r1_p05 b_r1_p2 b_r2_p05 b_r2_p1 b_r2_p2 site_rot05 site_rot2 site_dir05 site_dir2 extent05 extent2 nosmooth top10"}
DATASETS=${DATASETS:-"7scenes co3d"}
SCENES7=${SCENES7:-"chess fire heads office pumpkin redkitchen stairs"}
for v in $VARIANTS; do
  extra=${FLAGS[$v]}; [ -z "${FLAGS[$v]+x}" ] && { step "unknown variant $v"; continue; }
  for ds in $DATASETS; do
    out=$ROOT/$ds/$v; mkdir -p $out
    if [ $ds = 7scenes ]; then
      for s in $SCENES7; do GT=data/gt/7scenes_${s}_seq01.npz; [ -f $GT ] || continue
        [ -f $out/$s.json ] && [ -f $out/$s.npz ] && continue
        step "$v / 7scenes $s"
        python experiments/pilot_a.py --gt $GT --backbone ${SLAMBB:-vggt_omega} --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || { step "  FAILED: $(tail -n 2 $out/$s.log | tr '\n' ' ')"; }
      done
    else
      list=$(ls data/gt/co3d_full/*.npz | { [ -n "${ALL:-}" ] && cat || head -${NSEQ:-12}; })
      for GT in $list; do s=$(basename $GT .npz)
        [ -f $out/$s.json ] && [ -f $out/$s.npz ] && continue
        step "$v / co3d $s"
        python experiments/pilot_a.py --gt $GT --backbone ${BB:-vggt} --keyframe-stride 1 --max-frames 200 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || { step "  FAILED: $(tail -n 2 $out/$s.log | tr '\n' ' ')"; }
      done
    fi
  done
done
step "done -> python scripts/gate_table.py --root $ROOT --gt-dir data/gt --md outputs/gate_sweep.md"
