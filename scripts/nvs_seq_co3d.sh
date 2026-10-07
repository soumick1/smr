#!/bin/bash
# CO3Dv2 as the test-only NVS set: caches for the 37 validated orbits (every 8th frame held out) per backbone on 3 GPUs,
# then evaluation with the 7-Scenes-trained decoders (zero-shot, as Objaverse->GSO before).
#   bash scripts/nvs_seq_co3d.sh                      # six backbones, ~3 h on three GPUs
#   BBS="vggt_omega vggt" bash scripts/nvs_seq_co3d.sh
cd "$(dirname "$0")/.."; source .venv/bin/activate
BBS=${BBS:-"vggt_omega vggt pi3 stream3r streamvggt fast3r"}; GPUS=(0 1 2); CK=ckpt/nvsseq
step() { echo "[$(date +%H:%M:%S)] $*"; }
run_queue() { for i in 0 1 2; do ( n=0; while IFS= read -r cmd; do [ $((n % 3)) -eq $i ] && { step "GPU${GPUS[$i]}: $cmd" | cut -c1-160; CUDA_VISIBLE_DEVICES=${GPUS[$i]} bash -c "$cmd"; }; n=$((n+1)); done < $1 ) & done; wait; }
Q=/tmp/nvsq_co3d_$$.txt; : > $Q
for bb in $BBS; do mkdir -p cache/nvsseq/$bb/co3d_test outputs/nvs_seq_v2_co3d/$bb
  for gt in data/gt/co3d_full/*.npz; do s=$(basename $gt .npz); [ -f cache/nvsseq/$bb/co3d_test/$s/meta.json ] && continue
    echo "python experiments/nvs_seq_cache.py --gt $gt --backbone $bb --keyframe-stride 1 --max-frames 200 --holdout 8 --out cache/nvsseq/$bb/co3d_test/$s > cache/nvsseq/$bb/co3d_test/$s.log 2>&1 || echo CACHE-FAILED $bb $s" >> $Q
  done; done
step "stage 1: $(wc -l < $Q) CO3D cache jobs"; run_queue $Q
: > $Q
for bb in $BBS; do echo "python experiments/nvs_seq_decoder.py eval --caches 'cache/nvsseq/$bb/co3d_test/*' --ckpt-raw $CK/${bb}_raw.pt --ckpt-smr $CK/${bb}_smr.pt --methods raw smr gt --out outputs/nvs_seq_v2_co3d/$bb --save-images 4 2>&1 | grep -vi warn > outputs/nvs_seq_v2_co3d/$bb/eval.log" >> $Q; done
step "stage 2: $(wc -l < $Q) evaluations"; run_queue $Q
for bb in $BBS; do echo "== $bb"; tail -n 6 outputs/nvs_seq_v2_co3d/$bb/eval.log; done
step "ALL DONE"
