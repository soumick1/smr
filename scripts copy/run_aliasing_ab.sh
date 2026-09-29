#!/usr/bin/env bash
# A/B on the aliasing failures (office x6, TUM floor, TUM room) for the three
# fast backbones: v75 verification rules vs v74 behaviour vs --remeasure.
# Chunk passes are cached, so this is minutes per line.
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
mkdir -p outputs/reports/ab
for bb in vggt_omega vggt pi3; do
  for spec in "data/gt/7scenes_office_s0106.npz 10 16" "data/gt/tum_fr1_floor.npz 3 32" "data/gt/tum_fr1_room.npz 3 32"; do
    set -- $spec; gt=$1; st=$2; c=$3; scene=$(basename "$gt" .npz)
    for variant in "v75:" "v74:--no-mutual-nn --site-agree 1e9,1e9" "remeasure:--remeasure"; do
      name=${variant%%:*}; flags=${variant#*:}
      python experiments/pilot_a.py --gt "$gt" --backbone "$bb" --keyframe-stride $st --chunk $c --overlap $((c/2)) \
          --sites 2 --revisit-gap 32 --rows chained,smr,smr_pgo $flags \
          --json outputs/reports/ab/${scene}_${bb}_${name}.json 2>&1 \
          | grep -E "^(chained|smr )" | sed "s/^/$scene $bb $name  /"
    done
  done
done
