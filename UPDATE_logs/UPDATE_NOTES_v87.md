# smr_updates87 — dense pose-graph edges (smr_pgo_dense) + KITTI

    scp/unzip; python -m pytest -q    # expect 167 passed, 23 skipped
    git add -A && git commit -m "v87: dense 3D-3D pose-graph edges; KITTI prep" && git push

## TUM fr1: the VGGT-SLAM++ fight (their avg 0.036; we currently win 2/9)
Every junction/loop edge is re-measured from thousands of pixel-exact
3D-3D pairs (the shared frames unprojected in both chunks' own passes;
loops get one fresh mini-pass over the spatially closest frames), then the
same robust batch solve.  Cached chunk passes make the pose rows free; the
dense row adds one depth pass per chunk + one per loop.

    for s in 360 desk desk2 floor plant room rpy teddy xyz; do
      CUDA_VISIBLE_DEVICES=0 python experiments/pilot_a.py --gt data/gt/tum_fr1_$s.npz \
        --backbone vggt_omega --keyframe-stride 3 --chunk 32 --overlap 16 --sites 2 \
        --revisit-gap 32 --rows chained,smr,smr_pgo,smr_pgo_dense \
        --json outputs/reports/pilotA_tum_fr1_${s}_vggt_omega_dense.json 2>&1 | tee -a outputs/reports/tum_dense.log
    done

Predictions (recorded): dense halves junction noise on the short loop-free
sequences -> desk 0.042->0.020-0.028 (win vs 0.025 possible), desk2
0.038->0.022-0.030 (win vs 0.027 possible), plant 0.058->0.030-0.040,
360 0.082->0.045-0.060, teddy 0.063->0.035-0.045, room 0.091->0.050-0.070
(still lose vs 0.027), floor conceded (their DEM genuinely helps planar).
Target: 4-6 wins of 9 vs VGGT-SLAM++'s row.

## KITTI (bar: VGGT-Long cs60 avg* 19.30 excl seq 01; VGGT-SLAM++ 64.9)
Download odometry color (~65 GB) + poses into ~/kitti_odometry
(sequences/XX/image_2, poses/XX.txt), then:

    for q in 00 01 02 03 04 05 06 07 08 09 10; do
      python scripts/kitti_gt_poses.py --root ~/kitti_odometry --seq $q --out data/gt/kitti_$q.npz
    done
    for q in 03 04 07 06 05 09 10 00 02 08 01; do    # short first, long later
      CUDA_VISIBLE_DEVICES=0 python experiments/pilot_a.py --gt data/gt/kitti_$q.npz \
        --backbone vggt_omega --keyframe-stride 3 --chunk 32 --overlap 16 --sites 2 \
        --revisit-gap 64 --rows chained,smr,smr_pgo \
        --json outputs/reports/pilotA_kitti_${q}_vggt_omega.json 2>&1 | tee -a outputs/reports/kitti.log
    done

First-run notes: watch seq 03/04 (short, loop-free) -- chained there tells
us raw outdoor junction quality before anything else matters; 07/00/05/06/09
have loops (our closures + robust batch are the weapon); 08's revisits are
opposite-direction (DINO will miss them, as VGGT-Long's VPR does).  If 03/04
land under ~5 m, run the rest; predictions for the looped set: beat
VGGT-Long on >=3 of {00,02,05,06,07,09}.  Dense row on KITTI after TUM
confirms it (costs ~1 pass/chunk extra).
