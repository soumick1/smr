#!/usr/bin/env bash
# The standard protocol (MASt3R-SLAM -> VGGT-SLAM -> ViSTA-SLAM / SLAM-Former):
# 7-Scenes, seq-01 of all seven scenes; TUM RGB-D fr1, nine sequences.
# Downloads what is missing, extracts GT into data/gt/ with the convention
# measured once (c2w on both datasets; re-check with --diagnose-convention
# if a scene looks wrong).  Idempotent.
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
SEVEN=${SEVEN:-~/7scenes}; TUM=${TUM:-~/tum}
BASE=http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8
mkdir -p "$SEVEN" "$TUM" data/gt
for sc in chess fire heads office pumpkin redkitchen stairs; do
  if [ ! -d "$SEVEN/$sc/seq-01" ]; then
    (cd "$SEVEN" && wget -c -q --show-progress $BASE/$sc.zip && unzip -q -o $sc.zip && cd $sc && unzip -q -o seq-01.zip)
  fi
  [ -f data/gt/7scenes_${sc}_seq01.npz ] || \
    python scripts/indoor_gt_poses.py sevenscenes --root "$SEVEN" --scene $sc --seq 1 \
        --convention c2w --out data/gt/7scenes_${sc}_seq01.npz
done
for s in 360 desk desk2 floor plant room rpy teddy xyz; do
  d=rgbd_dataset_freiburg1_$s
  if [ ! -d "$TUM/$d" ]; then
    (cd "$TUM" && wget -c -q --show-progress https://cvg.cit.tum.de/rgbd/dataset/freiburg1/$d.tgz && tar xzf $d.tgz)
  fi
  [ -f data/gt/tum_fr1_$s.npz ] || \
    python scripts/indoor_gt_poses.py tum --root "$TUM" --sequence $d --convention c2w --out data/gt/tum_fr1_$s.npz
done
ls data/gt/
