#!/bin/bash
# Host RAM, CPU time and GPU utilisation for Table (compute) at N = 100 on one CO3D orbit, cache disabled so every pass runs.
#   GPU=0 bash scripts/resource_sweep.sh                       # fast six backbones (~10 min)
#   GPU=0 BBS="dust3r mast3r" bash scripts/resource_sweep.sh   # pairwise models (~30-40 min each, raw + smr)
#   python scripts/resource_summary.py --table                 # everything measured so far
cd "$(dirname "$0")/.."; source .venv/bin/activate; export GPU=${GPU:-0}; export CUDA_VISIBLE_DEVICES=$GPU
SEQ=${SEQ:-$(ls data/gt/co3d_full/*.npz | head -1)}; N=${N:-100}; STRIDE=$((200 / N)); mkdir -p outputs/resource /tmp/rs_cache
for bb in ${BBS:-"vggt vggt_omega pi3 fast3r streamvggt stream3r"}; do
  echo "== $bb (raw, windowed)"; rm -f /tmp/rs_cache/${bb}_raw*
  bash scripts/resource_sample.sh ${bb}_raw python experiments/pilot_a.py --gt $SEQ --backbone $bb --keyframe-stride $STRIDE --chunk 32 --overlap 16 --sites 2 \
      --rows chained --cache /tmp/rs_cache/${bb}_raw.npy --json outputs/resource/${bb}_raw.json
  echo "== $bb (+SMR, windowed)"; rm -f /tmp/rs_cache/${bb}_smr*
  bash scripts/resource_sample.sh ${bb}_smr python experiments/pilot_a.py --gt $SEQ --backbone $bb --keyframe-stride $STRIDE --chunk 32 --overlap 16 --sites 2 \
      --rows chained,smr --cache /tmp/rs_cache/${bb}_smr.npy --json outputs/resource/${bb}_smr.json
  echo "== $bb (native single pass, N=$N)"
  bash scripts/resource_sample.sh ${bb}_native python scripts/flops_probe.py --gt $SEQ --backbones $bb --n-list $N --redo --json /tmp/rs_cache/probe_${bb}.json
done
echo; python scripts/resource_summary.py --table
