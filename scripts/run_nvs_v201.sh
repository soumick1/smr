#!/bin/bash
# sequence NVS on pumpkin and office with the v201 protocol; logs to outputs/nvs_seq_<scene>.log
cd ~/smr && source .venv/bin/activate
for s in pumpkin office; do
  CUDA_VISIBLE_DEVICES=${GPU:-0} python experiments/nvs_sequence.py --gt data/gt/7scenes_${s}_seq01.npz --backbone vggt_omega --keyframe-stride 5 \
    --K 585,585,320,240 --head outputs/nvs/vggt_omega/ckpt_best.pt --out outputs/nvs_seq/7scenes_$s --save-images 6 2>&1 | grep -vi warn | tee outputs/nvs_seq_$s.log
done
echo NVS DONE
