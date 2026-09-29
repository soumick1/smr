#!/bin/bash
# Held-out Objaverse-LVIS rows (the training distribution) for every trained head: raw (reads=1) and read (reads=4).
#   DATA=~/data/nvs/objaverse_c70 CUDA_VISIBLE_DEVICES=0 bash scripts/nvs/eval_val_all.sh     (~35 min)
DATA=${DATA:-~/data/nvs/objaverse_c70}
for bb in ${BACKBONES:-vggt_omega vggt pi3 stream3r}; do
  CK=outputs/nvs/$bb/ckpt_best.pt; [ -f $CK ] || CK=outputs/nvs/$bb/ckpt_last.pt; [ -f $CK ] || { echo "no checkpoint for $bb"; continue; }
  for r in ${READS:-1 4}; do
    J=outputs/nvs/$bb/val_reads$r.jsonl
    for attempt in $(seq 1 20); do
      python experiments/eval_gso.py --backbone $bb --ckpt $CK --gso $DATA --val-only --reads $r --json $J >> outputs/reports/nvs_val_$bb.log 2>&1 && break
      if [ -f ${J%.jsonl}.inflight ]; then id=$(cat ${J%.jsonl}.inflight); echo "$id" >> ${J%.jsonl}.crashed; rm -f ${J%.jsonl}.inflight; echo "$bb reads=$r died on $id -> resuming"; else sleep 5; fi
    done
  done
done
python experiments/eval_gso.py --summary outputs/nvs/*/val_reads*.jsonl | tee outputs/nvs/summary_val.txt
