#!/bin/bash
# 7-Scenes on the dataset's STANDARD TEST SPLIT (TestSplit.txt of every scene), Table-2 configuration
# (keyframe stride 5, chunk 32 / overlap 16, 2 sites), rows chained / +SMR / +SMR+PGO.  Also runs the same
# trajectories with --index flat (the key-value baseline of App. L) so the two questions -- protocol and scaffold --
# are answered on the same passes.  Sequence zips are extracted on demand; GT npz files are written to data/gt/.
#   SEVEN=~/data/7scenes CUDA_VISIBLE_DEVICES=0 bash scripts/run_7scenes_testsplit.sh       (~1-1.5 h for VGGT-Omega)
#   BB=vggt_omega INDEXES="template flat" SCENES="chess fire ..." to restrict
cd "$(dirname "$0")/.."; source .venv/bin/activate
SEVEN=${SEVEN:-$HOME/data/7scenes}; BB=${BB:-vggt_omega}; OUT=${OUT:-outputs/7scenes_test}; mkdir -p $OUT outputs/reports
step() { echo "[$(date +%H:%M:%S)] $*"; }
for sc in ${SCENES:-chess fire heads office pumpkin redkitchen stairs}; do
  split=$SEVEN/$sc/TestSplit.txt
  [ -f $split ] || { step "$sc: no $split (is the scene zip extracted?)"; continue; }
  for seqname in $(tr -d '\r' < $split | sed 's/sequence//' | grep -o '[0-9]\+' ); do
    n=$(printf "%02d" $seqname); GT=data/gt/7scenes_${sc}_seq$n.npz
    if [ ! -d $SEVEN/$sc/seq-$n ]; then step "$sc seq-$n: extracting"; (cd $SEVEN/$sc && unzip -q -o seq-$n.zip) || { step "  missing seq-$n.zip"; continue; }; fi
    [ -f $GT ] || python scripts/indoor_gt_poses.py sevenscenes --root $SEVEN --scene $sc --seq $seqname --convention c2w --out $GT > $OUT/gt_${sc}_$n.log 2>&1 || { step "  GT failed: $(tail -n 1 $OUT/gt_${sc}_$n.log)"; continue; }
    for idx in ${INDEXES:-template flat}; do
      J=$OUT/${sc}_seq${n}_${BB}_$idx.json
      if [ -f $J ]; then   # sanity: a report whose 10-frame reference pass scores < 20 AUC came from a stale cache (v170 bug); redo it
        python - "$J" <<'PY2' && continue
import json, sys; d = json.load(open(sys.argv[1])); ref = d.get("reference") or {}
sys.exit(0 if ref.get("auc30", 100) >= 20 else 1)
PY2
        echo "[$(date +%H:%M:%S)]   $J is stale (reference AUC < 20) -> re-running"; rm -f $J
      fi
      step "$sc seq-$n ($BB, index=$idx)"
      python experiments/pilot_a.py --gt $GT --backbone $BB --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 --index $idx \
         --rows chained,smr,smr_pgo --json $J > ${J%.json}.log 2>&1 || step "  FAILED: $(tail -n 1 ${J%.json}.log)"
    done
  done
done
python scripts/sevenscenes_table.py --root $OUT --bb $BB
