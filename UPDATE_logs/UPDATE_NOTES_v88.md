# smr_updates88 — winner package: outdoor closure gate, LBA-lite, Virtual KITTI

    scp/unzip; python -m pytest -q    # expect 168 passed, 23 skipped
    git add -A && git commit -m "v88: outdoor gate cfg, smr_pgo_lba skip edges, vkitti prep, overlap>chunk/2 fix" && git push

Contains one latent-bug fix in the stitcher (overlap > chunk/2 crashed on an
unbound loop_w_scale; overlap-sourced edges are now always sequential) --
needed by the LBA runs below.  Dense placement/edges: REFUTED twice (TUM
edges, reloc), recorded; the rows are retired.

## 1. KITTI: fix the four wrong-closure sequences (bar: VGGT-Long avg* 19.30)
The failures were ACCEPTED closures 140-260 m from truth.  Gate them with
the existing consensus knobs (no new core code):

    for q in 02 05 07 08 09; do
      CUDA_VISIBLE_DEVICES=0 python experiments/pilot_a.py --gt data/gt/kitti_$q.npz \
        --backbone vggt_omega --keyframe-stride 3 --chunk 32 --overlap 16 --sites 2 \
        --revisit-gap 64 --site-agree 3,0.15 --mutual-nn --desc-thresh 0.6 \
        --rows chained,smr,smr_pgo \
        --json outputs/reports/pilotA_kitti_${q}_vggt_omega_gate.json 2>&1 | tee -a outputs/reports/kitti_gate.log
    done

Predictions (recorded): the tight agreement kills the wrong closures ->
02/05/07/08 land at chained level or better (85/43/42/54), 09 keeps its
good pgo (20.8); with the unchanged wins the table avg* becomes ~15-17
vs VGGT-Long 19.30.  If 05/07 still accept bad loops, next knob is
--revisit-gap 128.

## 2. TUM: no rerun -- the LBA-lite lever is REFUTED in simulation
Redundant skip edges (chunks c,c+2/c+3 fitted from cached poses at overlap
24) measured WORSE than smr_pgo across three seeds on the simulated world
(0.166/0.194/0.131 vs 0.105/0.153/0.082), scale-gated or not: long-span
pose fits inherit the same correlated per-pass distortion that sank the
dense edges.  The row exists behind --rows smr_pgo_lba for the record; do
not spend GPU on it.  TUM stays as measured: best-row avg 0.050 (w/o floor
0.045), wins 3/9 vs 2.0 and 2/9 vs SLAM++, and the paper says why (their
per-window LBA solves exactly this; ours deliberately never re-optimises
inside the backbone's window).

## 3. Virtual KITTI (bar: VGGT-Long 2.05 all-avg, uncalibrated block)
Download (~15 GB), prep, run (30 clips, ~2-3 GPU-h total):

    mkdir -p ~/vkitti && cd ~/vkitti
    wget -c https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_rgb.tar
    wget -c https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_extrinsicsgt.tar.gz
    tar xf vkitti_1.3.1_rgb.tar && tar xzf vkitti_1.3.1_extrinsicsgt.tar.gz && rm vkitti_1.3.1_rgb.tar
    # (if the host 404s: the same two files are linked from the official
    #  Naver Labs Europe 'Virtual KITTI 1' page -- 'virtual-kitti-1.3.1')

    cd ~/smr && source .venv/bin/activate
    for w in 0001 0002 0006 0018 0020; do for v in clone fog morning overcast rain sunset; do
      python scripts/vkitti_gt_poses.py --root ~/vkitti --world $w --var $v \
          --out data/gt/vkitti_${w}_${v}.npz
      CUDA_VISIBLE_DEVICES=0 python experiments/pilot_a.py --gt data/gt/vkitti_${w}_${v}.npz \
        --backbone vggt_omega --keyframe-stride 2 --chunk 32 --overlap 16 --sites 2 \
        --revisit-gap 64 --site-agree 3,0.15 --mutual-nn \
        --rows chained,smr,smr_pgo \
        --json outputs/reports/pilotA_vkitti_${w}_${v}_vggt_omega.json 2>&1 | tee -a outputs/reports/vkitti.log
    done; done

Predictions: clips are short loop-poor drives = exactly where our junctions
already beat VGGT-Long (KITTI 01/04/10); all-avg target < 2.0 with the risk
flagged as synthetic-weather domain shift for Omega/DINO (watch the
clone-vs-fog spread on world 0001 first).
