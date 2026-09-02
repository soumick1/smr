#!/bin/bash
# Build + run the OFFICIAL ETH3D evaluator on a predicted .ply.
#   bash scripts/eth3d_eval.sh outputs/points/eth3d_courtyard/fused.ply \
#        ~/data/eth3d/courtyard/dslr_scan_eval/scan_clean.mlp
set -e
TOOL=third_party/multi-view-evaluation/build/ETH3DMultiViewEvaluation
if [ ! -f "$TOOL" ]; then
  git clone -q https://github.com/ETH3D/multi-view-evaluation third_party/multi-view-evaluation || true
  cmake -S third_party/multi-view-evaluation -B third_party/multi-view-evaluation/build -DCMAKE_BUILD_TYPE=Release
  make -C third_party/multi-view-evaluation/build -j8
fi
"$TOOL" --reconstruction_ply_path "$1" --ground_truth_mlp_path "$2" --tolerances 0.02,0.05
