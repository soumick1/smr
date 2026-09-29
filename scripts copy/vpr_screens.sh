#!/bin/bash
# Launch the remaining VPR sweep in three detached screens (one GPU each). Safe to re-run: finished runs are skipped.
#   bash scripts/vpr_screens.sh          # start
#   screen -ls; tail -n 2 outputs/vpr_screen*.log        # monitor without attaching
#   grep -l "DONE" outputs/vpr_screen*.log | wc -l        # 3 = finished; then: bash scripts/vpr_collect.sh
# Split: GPU0 = 7-Scenes (both modes); GPU1 = CO3D anchored remainder + first half of plain arms; GPU2 = CO3D second half
# of plain arms in reverse scene order (no two screens share a dataset AND direction, so cache races are avoided).
cd ~/smr || exit 1
mkdir -p outputs
screen -dmS vpr0 bash -c 'cd ~/smr; { CUDA_VISIBLE_DEVICES=0 DATASETS=7scenes MODES=anchored ARMS="dino+salad@4096" bash scripts/vpr_sweep.sh; CUDA_VISIBLE_DEVICES=0 DATASETS=7scenes MODES=plain bash scripts/vpr_sweep.sh; echo VPR0 DONE; } 2>&1 | tee outputs/vpr_screen0.log'
screen -dmS vpr1 bash -c 'cd ~/smr; { CUDA_VISIBLE_DEVICES=1 DATASETS=co3d MODES=anchored ARMS="dino+salad@4096" bash scripts/vpr_sweep.sh; CUDA_VISIBLE_DEVICES=1 DATASETS=co3d MODES=plain ARMS="dino05 dino eigenplaces cosplace salad@4096" bash scripts/vpr_sweep.sh; echo VPR1 DONE; } 2>&1 | tee outputs/vpr_screen1.log'
screen -dmS vpr2 bash -c 'cd ~/smr; { CUDA_VISIBLE_DEVICES=2 DATASETS=co3d MODES=plain REVERSE=1 ARMS="boq@4096 netvlad mixvpr dino+eigenplaces dino+salad@4096" bash scripts/vpr_sweep.sh; echo VPR2 DONE; } 2>&1 | tee outputs/vpr_screen2.log'
sleep 2; screen -ls
echo "started; monitor with: tail -n 2 outputs/vpr_screen*.log"
