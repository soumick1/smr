#!/bin/bash
# Render the NVS training set on 3 GPUs (one shard each) and GSO on GPU 0 afterwards.
#   bash scripts/nvs/render_all.sh                    # 20K objaverse x 16 views + GSO x 14 views
#   NVIEWS=32 bash scripts/nvs/render_all.sh          # LVSM's 32 views per object
NVIEWS=${NVIEWS:-16}; DEV=${DEV:-OPTIX}
for g in 0 1 2; do
  screen -dmS ren$g bash -c "CUDA_VISIBLE_DEVICES=$g python scripts/nvs/render_bpy.py --list data/nvs/objaverse_index.json \
    --out ~/data/nvs/objaverse --protocol train --n-views $NVIEWS --device $DEV --shard $g --nshards 3 2>&1 | tee -a outputs/reports/nvs_render_gpu$g.log; \
    [ $g = 0 ] && CUDA_VISIBLE_DEVICES=0 python scripts/nvs/render_bpy.py --list data/nvs/gso_index.json --out ~/data/nvs/gso \
    --protocol gso --obj-up Z --device $DEV 2>&1 | tee -a outputs/reports/nvs_render_gso.log"
done
echo "screens ren0/ren1/ren2 started; progress: tail -n 2 outputs/reports/nvs_render_gpu*.log; failures: ls ~/data/nvs/objaverse/*/FAILED 2>/dev/null | wc -l"
