#!/usr/bin/env bash
# Drift-versus-length: the first N keyframes of one sequence for several N.
# Usage: bash scripts/run_length_sweep.sh data/gt/7scenes_chess_s0106.npz vggt
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
GT=${1:?gt npz}; BB=${2:-vggt}; STRIDE=${3:-10}
scene=$(basename "$GT" .npz)
for n in 100 200 300 450 600; do
  python experiments/pilot_a.py --gt "$GT" --backbone "$BB" --keyframe-stride $STRIDE \
      --chunk 16 --overlap 8 --sites 2 --max-frames $n --rows chained,smr,smr_pgo --ceiling \
      --json outputs/reports/len_${scene}_${BB}_n$n.json 2>&1 | grep -E "^(chained|smr|ceiling|    fits)"
done
python scripts/plot_length_sweep.py "outputs/reports/len_${scene}_${BB}_n*.json" \
    --out outputs/figures/length_sweep_${scene}_${BB}
