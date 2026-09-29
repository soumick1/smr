#!/bin/bash
# The second training variant the reviewer asked for: the head trained on the MEMORY'S geometry (the four-ordering read
# applied inside every training step) vs the shipped head trained on raw single-pass geometry.  Same everything else.
# Produces the 2x2 matrix {train raw, train +SMR} x {eval raw, eval +SMR} on GSO and held-out, and the training curves of
# both heads for fig_nvs_training.py --compare.
#   BB=vggt STEPS=10000 CUDA_VISIBLE_DEVICES=0 bash scripts/nvs/train_with_smr.sh   (~4 h train + ~2.5 h eval)
# The raw head is compared at the SAME step count: the shipped vggt head's metrics.csv is cut at STEPS in the plot, and
# its ckpt at STEPS comes from outputs/nvs/curve_<bb>/ckpt_step<STEPS>.pt if val_curve.sh ran, else ckpt_best (30K) is used
# and the caption says so.
cd "$(dirname "$0")/../.."; source .venv/bin/activate
BB=${BB:-vggt}; DATA=${DATA:-$HOME/data/nvs/objaverse_c70}; GSO=${GSO:-$HOME/data/nvs/gso_c70}; STEPS=${STEPS:-10000}
OUT=outputs/nvs/${BB}_trainsmr; mkdir -p $OUT
step() { echo "[$(date +%H:%M:%S)] $*"; }
if [ ! -f $OUT/ckpt_best.pt ]; then
  step "training $BB head on +SMR geometry (reads 4 in the loop), $STEPS steps"
  python experiments/train_nvs.py --backbone $BB --data $DATA --out $OUT --steps $STEPS --reads 4 --resume > $OUT/train_smr.log 2>&1 || { step "FAILED: $(tail -n 2 $OUT/train_smr.log)"; exit 1; }
fi
CK=$OUT/ckpt_best.pt
for r in 1 4; do
  J=$OUT/gso_reads$r.jsonl; [ -f $J ] && [ $(wc -l < $J) -ge 1033 ] || { step "eval GSO reads $r"; python experiments/eval_gso.py --backbone $BB --ckpt $CK --gso $GSO --reads $r --json $J > $OUT/eval_gso_r$r.log 2>&1 || step "  FAILED"; }
  J=$OUT/val_reads$r.jsonl; [ -f $J ] && [ $(wc -l < $J) -ge 174 ] || { step "eval held-out reads $r"; python experiments/eval_gso.py --backbone $BB --ckpt $CK --gso $DATA --val-only --reads $r --json $J > $OUT/eval_val_r$r.log 2>&1 || step "  FAILED"; }
done
python experiments/eval_gso.py --summary $OUT/gso_reads*.jsonl $OUT/val_reads*.jsonl outputs/nvs/$BB/gso_reads*.jsonl outputs/nvs/$BB/val_reads*.jsonl | tee $OUT/summary_2x2.txt
