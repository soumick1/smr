#!/usr/bin/env bash
# Pilot A sweep: every backbone x every prepared sequence, one process each.
# Usage: bash scripts/run_pilot_a_sweep.sh [backbones...]
# Prepared GT npz files are picked up from data/gt/*.npz; add more with
# scripts/indoor_gt_poses.py.  Passes are cached, so re-running is cheap.
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
BACKBONES=${*:-"vggt pi3 dust3r mast3r fast3r stream3r streamvggt monst3r vggt_omega"}
mkdir -p outputs/reports
for gt in data/gt/*.npz; do
  scene=$(basename "$gt" .npz)
  case "$scene" in
    tum_*) stride=5 ;;
    *)     stride=10 ;;
  esac
  for bb in $BACKBONES; do
    tag=outputs/reports/pilotA_${scene}_${bb}
    if [ -f "$tag.json" ]; then echo "skip $tag (exists)"; continue; fi
    echo "=== $scene / $bb (stride $stride) ==="
    python experiments/pilot_a.py --gt "$gt" --backbone "$bb" \
        --keyframe-stride $stride --chunk 16 --overlap 8 --sites 2 \
        --rows chained,smr,smr_pgo,classical --ceiling \
        --json "$tag.json" 2>&1 | tee "$tag.log" | grep -E "^(scene|  gate|    fits|method|ceiling|chained|smr|classical|  [a-z_]+ +vs chained|report)"
  done
done
python scripts/pilot_a_table.py outputs/reports/pilotA_*.json --tex outputs/reports/table_pilotA.tex
