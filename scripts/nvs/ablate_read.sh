#!/bin/bash
# In-window read ablations on GSO (VGGT head, first N objects, same objects for every variant; ~70 min for N=300).
#   CUDA_VISIBLE_DEVICES=0 bash scripts/nvs/ablate_read.sh
N=${N:-300}; BB=${BB:-vggt}; CK=outputs/nvs/$BB/ckpt_best.pt; [ -f $CK ] || CK=outputs/nvs/$BB/ckpt_last.pt
GSO=${GSO:-~/data/nvs/gso_c70}; mkdir -p outputs/nvs/ablate
run() { name=$1; shift; J=outputs/nvs/ablate/${BB}_$name.jsonl
  echo "[$(date +%H:%M:%S)] $name"; python experiments/eval_gso.py --backbone $BB --ckpt $CK --gso $GSO --limit $N --json $J "$@" > outputs/nvs/ablate/${BB}_$name.log 2>&1 || echo "  FAILED: $(tail -n 2 outputs/nvs/ablate/${BB}_$name.log | tr '\n' ' ')"; }
run reads1        --reads 1
run reads2        --reads 2
run reads4        --reads 4
run reads8        --reads 8
run reads4_noalign --reads 4 --no-align
run reads4_abstain --reads 4 --fallback none
python experiments/eval_gso.py --summary outputs/nvs/ablate/${BB}_*.jsonl | tee outputs/nvs/ablate/summary_$BB.txt
