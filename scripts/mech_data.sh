#!/bin/bash
# Data for the mechanism figures (saved trajectories + candidate logs), then the figures themselves.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/mech_data.sh          (~45 min: 23 uncached CO3D orbits; the rest is cached)
cd "$(dirname "$0")/.."; source .venv/bin/activate; mkdir -p outputs/est outputs/reports outputs/figures
step() { echo "[$(date +%H:%M:%S)] $*"; }
# 1. CO3D orbits, VGGT, N=200 (seams / funnel / revisits / ceilings)
for GT in data/gt/co3d_full/*.npz; do s=$(basename $GT .npz); [ -f outputs/est/mech_co3d_${s}_vggt.npz ] && continue
  step "co3d $s"; python experiments/pilot_a.py --gt $GT --backbone vggt --keyframe-stride 1 --max-frames 200 --chunk 32 --overlap 16 --sites 2 --ceiling \
     --rows chained,smr,smr_pgo --json outputs/reports/mech_co3d_${s}_vggt.json --save-est outputs/est/mech_co3d_${s}_vggt.npz > outputs/reports/mech_co3d_${s}_vggt.log 2>&1 || step "  FAILED"
done
# 2. 7-Scenes seq-01, VGGT-Omega, Table-2 configuration (seams / funnel)
for sc in chess fire heads office pumpkin redkitchen stairs; do GT=data/gt/7scenes_${sc}_seq01.npz; [ -f outputs/est/mech_7scenes_${sc}.npz ] && continue
  step "7scenes $sc"; python experiments/pilot_a.py --gt $GT --backbone vggt_omega --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 \
     --rows chained,smr,smr_pgo --json outputs/reports/mech_7scenes_${sc}.json --save-est outputs/est/mech_7scenes_${sc}.npz > outputs/reports/mech_7scenes_${sc}.log 2>&1 || step "  FAILED"
done
# 3. multi-session (timeline): the G4 configuration of run_all_gpu0.sh, streaming backbones
for GT in data/gt/7scenes_office_s0106.npz data/gt/7scenes_fire_s0104.npz; do s=$(basename $GT .npz)
  for bb in streamvggt stream3r; do [ -f outputs/est/mech_multi_${s}_$bb.npz ] && continue
    step "multi-session $s $bb"; python experiments/pilot_a.py --gt $GT --backbone $bb --keyframe-stride 10 --chunk 16 --overlap 8 --sites 2 --revisit-gap 32 \
       --rows chained,smr,smr_pgo --json outputs/reports/mech_multi_${s}_$bb.json --save-est outputs/est/mech_multi_${s}_$bb.npz > outputs/reports/mech_multi_${s}_$bb.log 2>&1 || step "  FAILED"
  done
done
# 4. figures
F=scripts/fig_mechanism.py
python $F seams    --est 'outputs/est/mech_co3d_*_vggt.npz' --out outputs/figures/mech_seams_co3d
python $F seams    --est 'outputs/est/mech_7scenes_*.npz'   --out outputs/figures/mech_seams_7scenes
python $F revisits --reports 'outputs/reports/mech_co3d_*_vggt.json' --out outputs/figures/mech_revisits          # add the RE10K / Table-1 reports via a wider glob when their names are known
python $F settling --out outputs/figures/mech_settling
python $F funnel   --reports 'outputs/reports/mech_co3d_*_vggt.json' --est 'outputs/est/mech_co3d_*_vggt.npz' --out outputs/figures/mech_funnel_co3d
python $F funnel   --reports 'outputs/reports/mech_7scenes_*.json'  --est 'outputs/est/mech_7scenes_*.npz'   --out outputs/figures/mech_funnel_7scenes
python $F flatbar  --root outputs/ablate --split outputs/7scenes_test --out outputs/figures/mech_flatbar
python $F nvshist  --nvs outputs/nvs/vggt --reads 4 --mark "11pro" "Jenga" "CONSTRUCTION" --out outputs/figures/mech_nvshist
for f in outputs/est/mech_multi_*.npz; do n=$(basename $f .npz); python $F timeline --est $f --title "${n#mech_multi_}" --out outputs/figures/${n/mech_multi_/mech_timeline_}; done
step "curves/gpumem need the Table-1 length reports: ls outputs/reports | grep -i 'len_\|n200' to find their glob, then"
step "  python $F curves --reports '<glob>' --out outputs/figures/mech_curves ; python $F gpumem --reports '<glob>' --out outputs/figures/mech_gpumem"
tar czf outputs/mech_figures.tgz outputs/figures/mech_*.png outputs/figures/mech_*.pdf
step "done: outputs/mech_figures.tgz"
