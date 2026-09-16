#!/bin/bash
# Validation curve RAW vs +SMR over training: re-run a head with intermediate checkpoints (seeded -> reproduces the shipped
# head), then evaluate every checkpoint on the 174 held-out objects with reads 1 and 4.
#   BB=vggt CUDA_VISIBLE_DEVICES=0 bash scripts/nvs/val_curve.sh        (~4 h train + 15 x ~9 min eval = ~6.5 h)
#   BB=vggt_omega CUDA_VISIBLE_DEVICES=1 bash scripts/nvs/val_curve.sh
cd "$(dirname "$0")/../.."; source .venv/bin/activate
BB=${BB:-vggt}; DATA=${DATA:-$HOME/data/nvs/objaverse_c70}; EVERY=${EVERY:-2000}; STEPS=${STEPS:-30000}
OUT=outputs/nvs/curve_$BB; mkdir -p $OUT
step() { echo "[$(date +%H:%M:%S)] $*"; }
if [ ! -f $OUT/ckpt_step$STEPS.pt ]; then
  step "training $BB with checkpoints every $EVERY"
  python experiments/train_nvs.py --backbone $BB --data $DATA --out $OUT --steps $STEPS --save-every $EVERY --resume > $OUT/train_curve.log 2>&1 || { step "training FAILED: $(tail -n 2 $OUT/train_curve.log)"; exit 1; }
fi
for ck in $(ls $OUT/ckpt_step*.pt | sort -t p -k3 -n); do s=$(basename $ck .pt); s=${s#ckpt_step}
  for r in 1 4; do J=$OUT/val_reads${r}_step$s.jsonl; [ -f $J ] && [ $(wc -l < $J) -ge 174 ] && continue
    step "$BB step $s reads $r"
    python experiments/eval_gso.py --backbone $BB --ckpt $ck --gso $DATA --val-only --reads $r --json $J > $OUT/eval_step${s}_r$r.log 2>&1 || step "  FAILED: $(tail -n 1 $OUT/eval_step${s}_r$r.log)"
  done
done
python experiments/eval_gso.py --summary $OUT/val_reads*_step*.jsonl | tee $OUT/summary_curve.txt
