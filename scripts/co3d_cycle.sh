#!/usr/bin/env bash
# CO3D on a tight disk: one category at a time --
#   download -> prune to images-only -> harvest pairs -> delete raw.
# Only the harvested npz pairs (a few GB/category) are kept.
#
#   bash scripts/co3d_cycle.sh hydrant vase teddybear apple bench plant
#
# Env overrides: CO3D_REPO (default: co3d), DATA (~/co3d_data),
#                OUT (data/completion_co3d), LIMIT (80 scenes/category)
set -e
CO3D_REPO=${CO3D_REPO:-co3d}
DATA=${DATA:-$HOME/co3d_full}
OUT=${OUT:-data/completion_co3d}
LIMIT=${LIMIT:-80}
BACKBONES=${BACKBONES:-vggt}
mkdir -p "$DATA"
for cat in "$@"; do
  echo "================ $cat : download ================"
  python "$CO3D_REPO/co3d/download_dataset.py" \
     --download_folder "$DATA" --download_categories "$cat" \
     --clear_archives_after_unpacking \
     --n_download_workers 4 --n_extract_workers 4
  echo "================ $cat : prune non-image payload ================"
  find "$DATA/$cat" -mindepth 2 -maxdepth 2 -type d \
       \( -name depths -o -name depth_masks -o -name masks \) \
       -exec rm -rf {} + 2>/dev/null || true
  find "$DATA/$cat" -type f \( -name "pointcloud.ply" -o -name "*.jgz" \) \
       -delete 2>/dev/null || true
  df -h "$DATA" | tail -1
  for bb in $BACKBONES; do
    echo "================ $cat : harvest [$bb] ================"
    python scripts/harvest_root.py --root "$DATA/$cat" --images-sub images \
       --backbone "$bb" --out "$OUT" --holdouts every:5 --context 1,4,all \
       --max-views 32 --limit-scenes "$LIMIT"
  done
  echo "================ $cat : delete raw ================"
  rm -rf "${DATA:?}/$cat"
  echo "$cat done; pairs so far: $(ls "$OUT" | wc -l)"
done
