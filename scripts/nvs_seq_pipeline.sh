#!/bin/bash
# Sequence NVS, end to end for one backbone (v202): caches on 3 GPUs, raw + SMR decoders in parallel, evaluation.
#   BB=vggt_omega bash scripts/nvs_seq_pipeline.sh            # full: ~25 min caches, ~1 h training, ~5 min eval
#   BB=vggt_omega SMOKE=1 bash scripts/nvs_seq_pipeline.sh    # 1 train + 1 test sequence, 60 steps: checks the whole path in ~10 min
# Train: every 7-Scenes sequence except seq-01 (all keyframes are inputs, targets drawn from them with neighbours excluded).
# Test:  the seven seq-01 trajectories (every 8th keyframe held out, never seen by the backbone).
cd "$(dirname "$0")/.."; source .venv/bin/activate
BB=${BB:-vggt_omega}; STEPS=${STEPS:-12000}; ROOTC=cache/nvsseq/$BB; OUT=outputs/nvs_seq_v2/$BB; CK=ckpt/nvsseq; mkdir -p $ROOTC/train $ROOTC/test $OUT $CK
step() { echo "[$(date +%H:%M:%S)] $*"; }
TRAIN=$(ls data/gt/7scenes_*_seq*.npz | grep -E '/7scenes_[a-z]+_seq[0-9]+\.npz$' | grep -v '_seq01\.npz$')
TEST=$(ls data/gt/7scenes_*_seq01.npz | grep -E '/7scenes_[a-z]+_seq01\.npz$')
if [ -n "${SMOKE:-}" ]; then TRAIN=$(echo "$TRAIN" | head -1); TEST=$(echo "$TEST" | grep pumpkin); STEPS=60; fi
[ -z "$TRAIN" ] && { echo "no training sequences: data/gt has no 7scenes_<scene>_seqNN.npz other than seq01"; exit 2; }
[ -z "$TEST" ] && { echo "no test sequences (7scenes_*_seq01.npz)"; exit 2; }
step "$BB: $(echo $TRAIN | wc -w) training sequences, $(echo $TEST | wc -w) test sequences, $STEPS steps"
cache_one() { gpu=$1; gt=$2; split=$3; ho=$4; s=$(basename $gt .npz | sed 's/^7scenes_//')
  [ -f $ROOTC/$split/$s/meta.json ] && return
  CUDA_VISIBLE_DEVICES=$gpu python experiments/nvs_seq_cache.py --gt $gt --backbone $BB --keyframe-stride 5 --K 585,585,320,240 \
     --holdout $ho --out $ROOTC/$split/$s > $ROOTC/$split/$s.log 2>&1 || step "  cache FAILED $s: $(grep -m1 -E 'Error|Traceback' $ROOTC/$split/$s.log)"; }
i=0; for gt in $TEST; do echo "$((i % 3)) $gt test 8"; i=$((i+1)); done > /tmp/nvsq_$BB.txt
for gt in $TRAIN; do echo "$((i % 3)) $gt train 0"; i=$((i+1)); done >> /tmp/nvsq_$BB.txt
for g in 0 1 2; do ( grep "^$g " /tmp/nvsq_$BB.txt | while read gpu gt split ho; do step "cache $split $(basename $gt)"; cache_one $gpu $gt $split $ho; done ) & done; wait
step "caches done: $(ls -d $ROOTC/train/*/ 2>/dev/null | wc -l) train, $(ls -d $ROOTC/test/*/ 2>/dev/null | wc -l) test"
for m in raw smr; do gpu=$([ $m = raw ] && echo 0 || echo 1)
  ( [ -f $CK/${BB}_$m.pt ] && [ -z "${SMOKE:-}" ] ) || CUDA_VISIBLE_DEVICES=$gpu python experiments/nvs_seq_decoder.py train --caches "$ROOTC/train/*" --method $m \
      --steps $STEPS --out $CK/${BB}_$m${SMOKE:+_smoke}.pt > $OUT/train_$m.log 2>&1 &
done; wait
step "training done: $(tail -n 2 $OUT/train_raw.log | head -1) / $(tail -n 2 $OUT/train_smr.log | head -1)"
CUDA_VISIBLE_DEVICES=0 python experiments/nvs_seq_decoder.py eval --caches "$ROOTC/test/*" --ckpt-raw $CK/${BB}_raw${SMOKE:+_smoke}.pt \
   --ckpt-smr $CK/${BB}_smr${SMOKE:+_smoke}.pt --out $OUT 2>&1 | grep -vi warn | tee $OUT/eval.log
step "done -> $OUT/eval.log"
