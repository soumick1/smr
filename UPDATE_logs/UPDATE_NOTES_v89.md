# smr_updates89 — wide-pass loop re-measurement (smr_pgo_wide)

    scp/unzip; python -m pytest -q     # expect 168 passed, 23 skipped

Accepted loop edges are re-measured with ONE wide joint pass (8+8 owned
frames of the two chunks, spatially closest under the current estimate),
pose-head fit only.  In simulation it TIES smr_pgo (recorded; the sim's
noise is baseline-independent so it cannot express the target error) --
the hypothesis is real-data wide-baseline measurement error, seen on 05
(true loops measured 4.9deg/14m made things worse).

    # 05 and 09 only (true loops); minutes each, passes cached
    for q in 05 09; do
      CUDA_VISIBLE_DEVICES=0 python experiments/pilot_a.py --gt data/gt/kitti_$q.npz \
        --backbone vggt_omega --keyframe-stride 3 --chunk 32 --overlap 16 --sites 2 \
        --revisit-gap 64 --site-agree 3,0.15 --mutual-nn --desc-thresh 0.6 \
        --rows chained,smr,smr_pgo,smr_pgo_wide \
        --json outputs/reports/pilotA_kitti_${q}_vggt_omega_wide.json 2>&1 | tee -a outputs/reports/kitti_wide.log
    done
    # 02: the gap is raw odometry (chained 85 vs VL 34, wrong loops unfixable
    # by consensus -- same-block lookalikes).  One junction-quality probe:
    CUDA_VISIBLE_DEVICES=0 python experiments/pilot_a.py --gt data/gt/kitti_02.npz \
      --backbone vggt_omega --keyframe-stride 2 --chunk 48 --overlap 24 --sites 2 \
      --revisit-gap 96 --site-agree 3,0.15 --mutual-nn --desc-thresh 0.6 \
      --rows chained,smr,smr_pgo \
      --json outputs/reports/pilotA_kitti_02_vggt_omega_k2c48.json 2>&1 | tee -a outputs/reports/kitti_wide.log

Predictions (recorded): 05 wide -> 12-20 (from 43 chained / 69 gated) if the
hypothesis holds, unchanged if not; 09 wide -> 8-11; 02 at stride2/chunk48 ->
chained 40-60 (denser junctions on the longest drive), still likely short of
VL's 34.  Table stance after this: wins 8-9/11, avg* ~19-21 vs VL 19.30 --
the average comes down to 02.
