#!/bin/bash
# Data + figures for the qualitative panels (GPU 0; ~30 min). Prints every step; never exits silently.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/fig_all.sh
cd "$(dirname "$0")/.." ; source .venv/bin/activate; mkdir -p outputs/est outputs/figures outputs/reports outputs/points
BB=${BB:-vggt}; SLAMBB=${SLAMBB:-vggt_omega}
step() { echo "[$(date +%H:%M:%S)] $*"; }
run() { local log=$1; shift; "$@" > "$log" 2>&1 && step "  ok" || { step "  FAILED (exit $?) -- tail of $log:"; tail -n 5 "$log" | sed 's/^/    /'; return 1; }; }
# skip_if <file> <log> <cmd...>: run only when <file> is missing (re-runs are cheap to resume)
skip_if() { local f=$1; shift; [ -f "$f" ] && { step "  exists: $f"; return 0; }; run "$@"; }

# ---- 1. camera pose vs length: two CO3D orbits at N=200
CO3D_DIR=""; for d in data/gt/co3d_N200 data/gt/co3d_full data/gt/co3d; do [ -d $d ] && ls $d/*.npz >/dev/null 2>&1 && { CO3D_DIR=$d; break; }; done
if [ -z "$CO3D_DIR" ]; then step "no CO3D GT directory with npz files found under data/gt/ (looked for co3d_N200, co3d_full, co3d)"; else
  step "CO3D GT from $CO3D_DIR: $(ls $CO3D_DIR/*.npz | wc -l) sequences"
  for GT in $(ls $CO3D_DIR/*.npz | head -${NCO3D:-2}); do s=$(basename $GT .npz)
    step "pose vs length: $s ($BB, N=200)"
    skip_if outputs/est/fig_${s}_$BB.npz outputs/reports/fig_${s}_$BB.log python experiments/pilot_a.py --gt $GT --backbone $BB --keyframe-stride ${CO3D_STRIDE:-1} --chunk 32 --overlap 16 --sites 2 --max-frames 200 \
        --rows chained,smr,smr_pgo --json outputs/reports/fig_${s}_$BB.json --save-est outputs/est/fig_${s}_$BB.npz || continue
    run outputs/figures/qual_${s}_$BB.log python scripts/fig_qualitative.py --est outputs/est/fig_${s}_$BB.npz --json outputs/reports/fig_${s}_$BB.json \
        --out outputs/figures/qual_${s}_$BB --title "$s, $BB, N=200"
    # rendered reconstruction with frusta by window (VGGT-style): passes fused under the raw and the +SMR junctions
    for row in chained smr; do
      step "  fused cloud of $s under the $row junctions"
      skip_if outputs/points/fig_${s}_${BB}_$row/fused.ply outputs/points/fig_${s}_${BB}_$row.log python experiments/points_suite.py --gt $GT --pilot outputs/est/fig_${s}_$BB.npz --row $row --backbone $BB --w 32 --overlap 16 --k-ctx 0 --stride 1 \
          --points-from auto --conf-abs 2.0 --fallback median --abstain-rel 0 --out-dir outputs/points/fig_${s}_${BB}_$row
    done
    [ -f outputs/points/fig_${s}_${BB}_chained/fused.ply ] && [ -f outputs/points/fig_${s}_${BB}_smr/fused.ply ] && \
      run outputs/figures/recon_${s}_$BB.log python scripts/fig_recon.py --est outputs/est/fig_${s}_$BB.npz --json outputs/reports/fig_${s}_$BB.json \
          --ply outputs/points/fig_${s}_${BB}_chained/fused.ply outputs/points/fig_${s}_${BB}_smr/fused.ply --rows chained smr \
          --labels "raw (chained Sim(3) windows)" "+SMR" --azimuth ${AZ:--12} --elev ${ELEV:-34} --dist ${DIST:-1.9} --out outputs/figures/recon_${s}_$BB
  done
fi
# ---- 2. SLAM on 7-Scenes: office and pumpkin (largest recoveries), trajectories + fused clouds
for s in ${SCENES:-office pumpkin}; do GT=""; for c in data/gt/7scenes_${s}_${SUFFIX:-all}.npz data/gt/7scenes_${s}_seq01.npz; do [ -f $c ] && { GT=$c; break; }; done
  [ -z "$GT" ] && { step "no GT npz for 7-Scenes $s"; continue; }; n=$(basename $GT .npz)
  step "SLAM: $n ($SLAMBB)"
  skip_if outputs/est/fig_${n}.npz outputs/reports/fig_${n}.log python experiments/pilot_a.py --gt $GT --backbone $SLAMBB --keyframe-stride ${STRIDE:-10} --chunk 32 --overlap 16 --sites 2 \
      --rows chained,smr,smr_pgo --json outputs/reports/fig_${n}.json --save-est outputs/est/fig_${n}.npz || continue
  run outputs/figures/qual_${n}.log python scripts/fig_qualitative.py --est outputs/est/fig_${n}.npz --json outputs/reports/fig_${n}.json --out outputs/figures/qual_${n} --title "7-Scenes $s"
  for row in chained smr; do
    step "  fused cloud under the $row junctions"
    skip_if outputs/points/fig_${n}_$row/fused.ply outputs/points/fig_${n}_$row.log python experiments/points_suite.py --gt $GT --pilot outputs/est/fig_${n}.npz --row $row --backbone $SLAMBB --w 32 --overlap 16 --k-ctx 0 --stride 1 \
        --points-from auto --conf-abs 2.0 --fallback median --abstain-rel 0 --out-dir outputs/points/fig_${n}_$row
  done
  [ -f outputs/points/fig_${n}_chained/fused.ply ] && [ -f outputs/points/fig_${n}_smr/fused.ply ] && \
    run outputs/figures/recon_${n}.log python scripts/fig_recon.py --est outputs/est/fig_${n}.npz --json outputs/reports/fig_${n}.json \
        --ply outputs/points/fig_${n}_chained/fused.ply outputs/points/fig_${n}_smr/fused.ply --rows chained smr \
        --labels "raw (chained Sim(3) windows)" "+SMR" --azimuth ${AZ:--12} --elev ${ELEV:-34} --dist ${DIST:-1.9} ${ZOOM:+--zoom $ZOOM} --out outputs/figures/recon_${n}
done
# ---- 3. NVS strips: 8 GSO objects, raw and read, then the montage
CK=outputs/nvs/vggt/ckpt_best.pt; [ -f $CK ] || CK=outputs/nvs/vggt/ckpt_last.pt
if [ -f $CK ] && [ -z "${SKIP_NVS:-}" ]; then
  for r in 1 4; do step "NVS images (reads=$r)"
    run outputs/nvs/vggt/fig_reads$r.log python experiments/eval_gso.py --backbone vggt --ckpt $CK --gso ~/data/nvs/gso_c70 --reads $r --limit 8 \
        --json outputs/nvs/vggt/fig_reads$r.jsonl --save-images outputs/nvs/vggt/images --save-n 8
  done
  objs=$(ls outputs/nvs/vggt/images 2>/dev/null | head -3 | tr '\n' ' ')
  [ -n "$objs" ] && run outputs/figures/nvs_vggt.log python scripts/fig_nvs_strip.py --dir outputs/nvs/vggt/images --objects $objs --target 2 --out outputs/figures/nvs_vggt
else step "no VGGT NVS checkpoint; skipping the NVS strip"; fi
step "figures: $(ls outputs/figures/*.pdf 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
