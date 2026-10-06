#!/bin/bash
# Sequence NVS (zero-shot decoders) on 7-Scenes seq-01 (VGGT-Omega) and the first 12 CO3D orbits (VGGT).
#   CUDA_VISIBLE_DEVICES=0 bash scripts/nvs_sequence_all.sh
cd "$(dirname "$0")/.."; source .venv/bin/activate; mkdir -p outputs/nvs_seq
step() { echo "[$(date +%H:%M:%S)] $*"; }
for s in ${SCENES7:-"pumpkin office chess fire heads redkitchen stairs"}; do
  [ -f outputs/nvs_seq/7scenes_$s/summary.json ] && continue; step "7scenes $s"
  python experiments/nvs_sequence.py --gt data/gt/7scenes_${s}_seq01.npz --backbone vggt_omega --keyframe-stride 5 --K 585,585,320,240 \
      --head outputs/nvs/vggt_omega/ckpt_best.pt --out outputs/nvs_seq/7scenes_$s --save-images 4 ${EXTRA:-} > outputs/nvs_seq/7scenes_$s.log 2>&1 || step "  FAILED: $(grep -m1 -E 'Error|Traceback' outputs/nvs_seq/7scenes_$s.log | cut -c1-160)"
done
for GT in $(ls data/gt/co3d_full/*.npz | head -${NSEQ:-12}); do s=$(basename $GT .npz)
  [ -f outputs/nvs_seq/co3d_$s/summary.json ] && continue; step "co3d $s"
  python experiments/nvs_sequence.py --gt $GT --backbone vggt --keyframe-stride 1 --max-frames 200 \
      --head outputs/nvs/vggt/ckpt_best.pt --out outputs/nvs_seq/co3d_$s --save-images 4 ${EXTRA:-} > outputs/nvs_seq/co3d_$s.log 2>&1 || step "  FAILED: $(grep -m1 -E 'Error|Traceback' outputs/nvs_seq/co3d_$s.log | cut -c1-160)"
done
python - <<'PY'
import glob, json, numpy as np
rows = [json.load(open(f)) for f in sorted(glob.glob("outputs/nvs_seq/*/summary.json"))]
for ds in ("7scenes", "co3d"):
    R = [r for r in rows if ds in r["gt"].lower() or (ds == "co3d" and "co3d" in r["gt"])]
    if not R: continue
    print(f"== {ds}: {len(R)} sequences, mean closures {np.mean([r['closures'] for r in R]):.1f}")
    for m in ("raw", "smr", "smr_pgo", "oracle"):
        v = [r["means"][m] for r in R if m in r["means"]]
        if v: print(f"  {m:<8} PSNR {np.mean([x['psnr'] for x in v]):.2f}  SSIM {np.mean([x['ssim'] for x in v]):.3f}  LPIPS {np.mean([x['lpips'] for x in v]):.3f}")
PY
