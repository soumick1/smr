#!/bin/bash
# All backbones on all GPUs: caches queued over 3 GPUs, then the 2 decoders per backbone queued 3 at a time, then evaluation.
#   bash scripts/nvs_seq_all.sh                                   # six backbones
#   BBS="vggt_omega vggt" bash scripts/nvs_seq_all.sh             # subset
cd "$(dirname "$0")/.."; source .venv/bin/activate
BBS=${BBS:-"vggt_omega vggt pi3 stream3r streamvggt fast3r"}; STEPS=${STEPS:-12000}; GPUS=(0 1 2); CK=ckpt/nvsseq; mkdir -p $CK
step() { echo "[$(date +%H:%M:%S)] $*"; }
TRAIN=$(ls data/gt/7scenes_*_seq*.npz | grep -E '/7scenes_[a-z]+_seq[0-9]+\.npz$' | grep -v '_seq01\.npz$')
TEST=$(ls data/gt/7scenes_*_seq01.npz | grep -E '/7scenes_[a-z]+_seq01\.npz$')
run_queue() { # $1 = file with one command per line; distributed round-robin over the GPUs, each GPU sequential
  for i in 0 1 2; do ( n=0; while IFS= read -r cmd; do [ $((n % 3)) -eq $i ] && { step "GPU${GPUS[$i]}: $cmd" | cut -c1-160; CUDA_VISIBLE_DEVICES=${GPUS[$i]} bash -c "$cmd"; }; n=$((n+1)); done < $1 ) & done; wait; }
Q=/tmp/nvsq_all_$$.txt; : > $Q
for bb in $BBS; do mkdir -p cache/nvsseq/$bb/train cache/nvsseq/$bb/test outputs/nvs_seq_v2/$bb
  for gt in $TEST $TRAIN; do s=$(basename $gt .npz | sed 's/^7scenes_//'); split=train; ho=0; echo "$gt" | grep -q '_seq01\.npz$' && { split=test; ho=8; }
    [ -f cache/nvsseq/$bb/$split/$s/meta.json ] && continue
    echo "python experiments/nvs_seq_cache.py --gt $gt --backbone $bb --keyframe-stride 5 --K 585,585,320,240 --holdout $ho --out cache/nvsseq/$bb/$split/$s > cache/nvsseq/$bb/$split/$s.log 2>&1 || echo CACHE-FAILED $bb $s" >> $Q
  done; done
step "stage 1: $(wc -l < $Q) cache jobs"; run_queue $Q
: > $Q
for bb in $BBS; do for m in raw smr; do [ -f $CK/${bb}_$m.pt ] && continue
  echo "python experiments/nvs_seq_decoder.py train --caches 'cache/nvsseq/$bb/train/*' --method $m --steps $STEPS --out $CK/${bb}_$m.pt > outputs/nvs_seq_v2/$bb/train_$m.log 2>&1 || echo TRAIN-FAILED $bb $m" >> $Q
done; done
step "stage 2: $(wc -l < $Q) decoder trainings"; run_queue $Q
: > $Q
for bb in $BBS; do echo "python experiments/nvs_seq_decoder.py eval --caches 'cache/nvsseq/$bb/test/*' --ckpt-raw $CK/${bb}_raw.pt --ckpt-smr $CK/${bb}_smr.pt --out outputs/nvs_seq_v2/$bb 2>&1 | grep -vi warn > outputs/nvs_seq_v2/$bb/eval.log" >> $Q; done
step "stage 3: $(wc -l < $Q) evaluations"; run_queue $Q
for bb in $BBS; do echo "== $bb"; tail -n 7 outputs/nvs_seq_v2/$bb/eval.log; done
step "ALL DONE"
