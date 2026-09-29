#!/bin/bash
# Train + evaluate the Gaussian heads of the remaining four backbones on the SAME protocol as the first four
# (train_nvs.py defaults: 1 object/step, 4 inputs -> 4 targets, AdamW 2e-4, 500-step warm-up + cosine; eval_gso.py:
# 4 inputs / 10 targets, reads 1 and 4, GSO 1,033 + held-out 174).  Step budgets: 30K for Fast3R / StreamVGGT (0.3-0.5
# s/step), 10K for DUSt3R / MASt3R (global alignment in every pass, 1.5-3 s/step; the first four heads were within 0.3 dB
# of final by 10K).  Each GPU runs its queue in one screen: smoke -> train -> GSO reads 1,4 -> held-out reads 1,4.
#   bash scripts/nvs/train_rest.sh            (launches 3 screens; ~12-14 h wall)
#   SMOKE_ONLY=1 bash scripts/nvs/train_rest.sh   (20-step smoke test of all four on GPU 0, ~10 min: run this FIRST)
cd "$(dirname "$0")/.."; source .venv/bin/activate
ENV='cd ~/smr && source .venv/bin/activate && export CUDA_HOME=$HOME/miniconda3 PATH=$PATH:$HOME/miniconda3/bin CPLUS_INCLUDE_PATH=$HOME/miniconda3/targets/x86_64-linux/include LD_LIBRARY_PATH=$HOME/miniconda3/targets/x86_64-linux/lib'
DATA=${DATA:-$HOME/data/nvs/objaverse_c70}; GSO=${GSO:-$HOME/data/nvs/gso_c70}; mkdir -p outputs/nvs outputs/reports
declare -A STEPS=([fast3r]=30000 [streamvggt]=30000 [dust3r]=10000 [mast3r]=10000)
if [ -n "${SMOKE_ONLY:-}" ]; then
  for bb in fast3r streamvggt dust3r mast3r; do echo "== smoke $bb"
    CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0} python experiments/train_nvs.py --backbone $bb --data $DATA --out outputs/nvs/smoke_$bb --smoke 2>&1 | grep -E "step 20|val@20|done:|Error|error" | tail -4
  done; exit 0
fi
queue() {  # queue <gpu> <bb...>
  local gpu=$1; shift; local cmd=""
  for bb in "$@"; do
    cmd="$cmd echo \"[\$(date +%H:%M:%S)] train $bb (${STEPS[$bb]} steps)\";"
    cmd="$cmd python experiments/train_nvs.py --backbone $bb --data $DATA --out outputs/nvs/$bb --steps ${STEPS[$bb]} --resume >> outputs/reports/nvs_train_$bb.log 2>&1;"
    cmd="$cmd echo \"[\$(date +%H:%M:%S)] eval $bb\"; READS='1 4' BACKBONES=$bb GSO=$GSO bash scripts/nvs/eval_all.sh >> outputs/reports/nvs_eval_$bb.log 2>&1;"
    cmd="$cmd READS='1 4' BACKBONES=$bb DATA=$DATA bash scripts/nvs/eval_val_all.sh >> outputs/reports/nvs_val_$bb.log 2>&1;"
    cmd="$cmd echo \"[\$(date +%H:%M:%S)] $bb done\";"
  done
  screen -dmS nvs_gpu$gpu bash -c "$ENV && export CUDA_VISIBLE_DEVICES=$gpu && $cmd echo ALL_DONE"
  echo "GPU $gpu: $*"
}
queue 0 fast3r streamvggt
queue 1 dust3r
queue 2 mast3r
echo "progress: tail -n 2 outputs/nvs/{fast3r,streamvggt,dust3r,mast3r}/train.log ; wc -l outputs/nvs/*/gso_reads*.jsonl outputs/nvs/*/val_reads*.jsonl"
