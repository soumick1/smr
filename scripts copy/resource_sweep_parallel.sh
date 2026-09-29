#!/bin/bash
# The resource sweep on three GPUs at once: two backbones per GPU, raw / +SMR / native each, then one summary table.
#   bash scripts/resource_sweep_parallel.sh                 # fast six (~8 min)
#   bash scripts/resource_sweep_parallel.sh pairwise        # dust3r on GPU 1, mast3r on GPU 2 (~40 min)
cd "$(dirname "$0")/.."; mkdir -p outputs/resource
if [ "$1" = pairwise ]; then
  GPU=1 BBS="dust3r" bash scripts/resource_sweep.sh > outputs/resource/gpu1_pairwise.log 2>&1 &
  GPU=2 BBS="mast3r" bash scripts/resource_sweep.sh > outputs/resource/gpu2_pairwise.log 2>&1 &
else
  GPU=0 BBS="vggt fast3r"        bash scripts/resource_sweep.sh > outputs/resource/gpu0.log 2>&1 &
  GPU=1 BBS="vggt_omega streamvggt" bash scripts/resource_sweep.sh > outputs/resource/gpu1.log 2>&1 &
  GPU=2 BBS="pi3 stream3r"       bash scripts/resource_sweep.sh > outputs/resource/gpu2.log 2>&1 &
fi
echo "started $(date +%H:%M:%S); progress: tail -n 3 outputs/resource/gpu*.log"
wait
echo "finished $(date +%H:%M:%S)"; source .venv/bin/activate; python scripts/resource_summary.py --table
