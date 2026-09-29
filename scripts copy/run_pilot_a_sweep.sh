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
  # strides of the stride-based systems in the VGGT-SLAM comparison tables
  # (ViSTA-SLAM, SLAM-Former): 5 on 7-Scenes, 3 on TUM.  Multi-session
  # concatenations keep 10 so they stay at 600 keyframes.
  # chunk 32 / overlap 16 on the standard sequences: measured on chess seq-01
  # (VGGT, stride 5) chained 0.0448 -> 0.0422 and smr+pgo 0.0407 against a
  # 0.0395 single pass, and it is VGGT-SLAM's window.  Multi-session keeps
  # chunk 16 (the drift curve already measured) -- add c32 there by hand.
  # Revisit gap fixed at 32 keyframes so the loop columns compare across chunk sizes.
  case "$scene" in
    tum_*)          stride=3;  chunk=32 ;;
    7scenes_*_s0*)  stride=10; chunk=16 ;;
    *)              stride=5;  chunk=32 ;;
  esac
  overlap=$((chunk/2))
  for bb in $BACKBONES; do
    tag=outputs/reports/pilotA_${scene}_${bb}_s${stride}_c${chunk}
    if [ -f "$tag.json" ]; then echo "skip $tag (exists)"; continue; fi
    free_gb=$(df -BG --output=avail "$PWD" | tail -1 | tr -d ' G')
    tmp_gb=$(df -BG --output=avail /tmp | tail -1 | tr -d ' G')
    if [ "$free_gb" -lt 5 ] || [ "$tmp_gb" -lt 5 ]; then
      echo "!!! under 5 GB free (repo: ${free_gb}G, /tmp: ${tmp_gb}G) -- stopping before $scene/$bb"; exit 2
    fi
    echo "=== $scene / $bb (stride $stride) ==="
    python experiments/pilot_a.py --gt "$gt" --backbone "$bb" \
        --keyframe-stride $stride --chunk $chunk --overlap $overlap --sites 2 --revisit-gap 32 \
        --rows chained,smr,smr_pgo,classical --ceiling \
        --json "$tag.json" 2>&1 | tee "$tag.log" | grep -E "^(scene|  gate|  \*\*\* GATE|    fits|method|ceiling|chained|smr|classical|  [a-z_]+ +vs chained|report|Traceback|[A-Za-z]*Error|  File )"
  done
done
python scripts/pilot_a_table.py outputs/reports/pilotA_*.json --tex outputs/reports/table_pilotA.tex
