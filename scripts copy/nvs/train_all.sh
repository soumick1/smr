#!/bin/bash
# Train one Gaussian head per backbone. Backbones are queued round-robin over the GPUs (4 heads on 3 GPUs:
# GPU0 runs the 1st and 4th sequentially). Each screen runs its queue in order; frozen backbone + head share a GPU.
#   DATA=~/data/nvs/objaverse_c70 bash scripts/nvs/train_all.sh
#   STEPS=20000 BACKBONES="vggt pi3" bash scripts/nvs/train_all.sh
STEPS=${STEPS:-30000}; DATA=${DATA:-~/data/nvs/objaverse_c70}; NGPU=${NGPU:-3}
read -ra BBS <<< "${BACKBONES:-vggt_omega vggt pi3 stream3r}"
declare -a Q; for g in $(seq 0 $((NGPU-1))); do Q[$g]=""; done
for i in "${!BBS[@]}"; do g=$((i % NGPU)); Q[$g]="${Q[$g]} ${BBS[$i]}"; done
for g in $(seq 0 $((NGPU-1))); do
  [ -z "${Q[$g]}" ] && continue
  CMD=""
  for bb in ${Q[$g]}; do
    CMD="$CMD CUDA_VISIBLE_DEVICES=$g python experiments/train_nvs.py --backbone $bb --data $DATA --out outputs/nvs/$bb --steps $STEPS --resume 2>&1 | tee -a outputs/reports/nvs_train_$bb.log;"
  done
  screen -dmS nvs_gpu$g bash -c "$CMD"
  echo "GPU $g queue:${Q[$g]}"
done
echo "progress: tail -n 2 outputs/nvs/*/train.log"
