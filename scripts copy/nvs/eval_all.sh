#!/bin/bash
# GSO rows for every trained head: raw geometry (reads=1) and the in-window read (reads=4) for ALL backbones.
# Crash-proof: if the evaluator process dies without a Python exception (e.g. a CUDA abort on one object), the object
# recorded in <json>.inflight is added to <json>.crashed and the row is resumed; up to 40 restarts per row.
#   GSO=~/data/nvs/gso_c70 bash scripts/nvs/eval_all.sh
GSO=${GSO:-~/data/nvs/gso_c70}
for bb in ${BACKBONES:-vggt_omega vggt pi3 stream3r}; do
  CK=outputs/nvs/$bb/ckpt_best.pt; [ -f $CK ] || CK=outputs/nvs/$bb/ckpt_last.pt; [ -f $CK ] || { echo "no checkpoint for $bb"; continue; }
  for r in ${READS:-1 4}; do
    J=outputs/nvs/$bb/gso_reads$r.jsonl
    for attempt in $(seq 1 40); do
      python experiments/eval_gso.py --backbone $bb --ckpt $CK --gso $GSO --reads $r --json $J >> outputs/reports/nvs_eval_$bb.log 2>&1 && break
      if [ -f ${J%.jsonl}.inflight ]; then id=$(cat ${J%.jsonl}.inflight); echo "$id" >> ${J%.jsonl}.crashed; rm -f ${J%.jsonl}.inflight
        echo "[$(date +%H:%M:%S)] $bb reads=$r: process died on '$id' (attempt $attempt) -> marked crashed, resuming" | tee -a outputs/reports/nvs_eval_$bb.log
      else echo "[$(date +%H:%M:%S)] $bb reads=$r: process died outside an object (attempt $attempt)" | tee -a outputs/reports/nvs_eval_$bb.log; sleep 5; fi
    done
  done
done
python experiments/eval_gso.py --summary outputs/nvs/*/gso_reads*.jsonl
