#!/bin/bash
# Fetch the ETH3D OFFICIAL ground-truth depth maps (rendered from the laser
# scan, occlusion + masks applied; the "official masks" of VGGT Sec. 4.3) and
# the distorted-camera calibration they are defined on (dslr_calibration_jpg,
# THIN_PRISM_FISHEYE), for the 13 high-res training scenes.
#   bash scripts/eth3d_fetch_gt.sh ~/data/eth3d            # ~10 GB download, ~4 GB kept
#   bash scripts/eth3d_fetch_gt.sh ~/data/eth3d --keep-jpg # also keep the distorted JPGs
# Archives extract as <scene>/ground_truth_depth/dslr_images/<name> (raw float32)
# and <scene>/dslr_calibration_jpg/{cameras,images,points3D}.txt next to the
# existing <scene>/dslr_calibration_undistorted and dslr_scan_eval folders.
set -e
ROOT=${1:?usage: eth3d_fetch_gt.sh <root with <scene>/ folders> [--keep-jpg]}; KEEP=${2:-}
command -v 7z >/dev/null || { echo "need p7zip: sudo apt-get install p7zip-full"; exit 1; }
SCENES="courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains"
mkdir -p "$ROOT/_archives"; cd "$ROOT/_archives"
for sc in $SCENES; do
  for kind in dslr_depth dslr_jpg; do
    f=${sc}_${kind}.7z
    [ -f "$f" ] || wget -q --show-progress --no-check-certificate "https://www.eth3d.net/data/$f"
    7z x -y -o"$ROOT" "$f" >/dev/null
  done
  [ -n "$KEEP" ] || rm -rf "$ROOT/$sc/images/dslr_images"   # only the calibration of the jpg archive is needed
  echo "$sc: $(ls "$ROOT/$sc/ground_truth_depth/dslr_images" | wc -l) GT depth maps; $(head -c 0 "$ROOT/$sc/dslr_calibration_jpg/cameras.txt" && echo calib ok)"
done
echo "done. Verify one file: python -c \"import numpy as np; d=np.fromfile('$ROOT/courtyard/ground_truth_depth/dslr_images/'+__import__('os').listdir('$ROOT/courtyard/ground_truth_depth/dslr_images')[0], dtype=np.float32); print(d.size, 6048*4032, np.isfinite(d).mean())\""
