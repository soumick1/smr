#!/bin/bash
# v180: window geometry from the plain pass vs the anchored pass, on fully cached passes (~10 s per sequence).
#   CUDA_VISIBLE_DEVICES=0 bash scripts/local_from_sweep.sh
# Arms (7-Scenes seq-01 x 7 with VGGT-Omega; CO3D 12 orbits with VGGT):
#   plain_seed0/1/2   --local-from plain --index template --seed N   (does the seed-1 chess blow-up disappear?)
#   plain_flat        --local-from plain --index flat
#   anch_none         --correction none                (anchored passes only; Table 6 "not applied online", explained)
#   plain_none        --correction none --local-from plain   (must equal raw exactly; test G)
# Then:
#   for v in plain_seed0 plain_seed1 plain_seed2 plain_flat; do python scripts/paired_stats.py reports \
#       --a 'outputs/ablate/seeds/7scenes/flat/*.json' --b "outputs/ablate/localfrom/7scenes/$v/*.json" --row smr --metric ate_rmse --per-item | tail -9; done
#   python scripts/paired_stats.py rows --reports 'outputs/ablate/localfrom/7scenes/plain_none/*.json' --row-a chained --row-b smr --metric ate_rmse   # delta 0
#   python scripts/closure_reliability.py --glob 'outputs/ablate/localfrom/7scenes/plain_seed1/*.json' --gt-dir data/gt \
#       --gt-pattern '7scenes={stem}_seq01.npz' --est-dir outputs/ablate/localfrom/7scenes/plain_seed1 --md outputs/rel_plain_seed1.md
cd "$(dirname "$0")/.."; source .venv/bin/activate; ROOT=${ROOT:-outputs/ablate/localfrom}; mkdir -p $ROOT
# refuse to run against an unpatched tree (every variant needs --local-from and --index)
python experiments/pilot_a.py --help 2>/dev/null | grep -q -- "--local-from" || { echo "pilot_a lacks --local-from: run  python scripts/apply_all_patches.py  first"; exit 2; }
grep -q "local_from" src/smr/stitch/anchored.py || { echo "anchored.py lacks local_from: run  python scripts/apply_all_patches.py  first"; exit 2; }
step() { echo "[$(date +%H:%M:%S)] $*"; }
declare -A FLAGS=(
  [plain_seed0]="--local-from plain --index template --seed 0"
  [plain_seed1]="--local-from plain --index template --seed 1"
  [plain_seed2]="--local-from plain --index template --seed 2"
  [plain_flat]="--local-from plain --index flat"
  [anch_none]="--correction none"
  [plain_none]="--correction none --local-from plain"
)
VARIANTS=${VARIANTS:-"plain_seed0 plain_seed1 plain_seed2 plain_flat anch_none plain_none"}; DATASETS=${DATASETS:-"7scenes co3d"}
SCENES7=${SCENES7:-"chess fire heads office pumpkin redkitchen stairs"}
for v in $VARIANTS; do extra=${FLAGS[$v]}; [ -z "${FLAGS[$v]+x}" ] && { step "unknown variant $v"; continue; }
  for ds in $DATASETS; do out=$ROOT/$ds/$v; mkdir -p $out
    if [ $ds = 7scenes ]; then
      for s in $SCENES7; do GT=data/gt/7scenes_${s}_seq01.npz; [ -f $GT ] || continue; [ -f $out/$s.json ] && [ -f $out/$s.npz ] && continue
        step "$v / 7scenes $s"
        python experiments/pilot_a.py --gt $GT --backbone ${SLAMBB:-vggt_omega} --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || step "  FAILED: $(tail -n 2 $out/$s.log | tr '\n' ' ')"
      done
    else
      list=$(ls data/gt/co3d_full/*.npz | { [ -n "${ALL:-}" ] && cat || head -${NSEQ:-12}; })
      for GT in $list; do s=$(basename $GT .npz); [ -f $out/$s.json ] && [ -f $out/$s.npz ] && continue
        step "$v / co3d $s"
        python experiments/pilot_a.py --gt $GT --backbone ${BB:-vggt} --keyframe-stride 1 --max-frames 200 --chunk 32 --overlap 16 --sites 2 $extra \
           --rows chained,smr,smr_pgo --json $out/$s.json --save-est $out/$s.npz > $out/$s.log 2>&1 || step "  FAILED: $(tail -n 2 $out/$s.log | tr '\n' ' ')"
      done
    fi
  done
done
step "done"
