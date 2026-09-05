# smr_updates69 — the standard protocol (all 7-Scenes + all TUM fr1), VGGT-SLAM positioning

    scp -P 22 /mnt/c/MyFiles/smr_updates/smr_updates69.zip soumick@10.97.144.63:~/
    ssh -p 22 soumick@10.97.144.63 "cd ~/smr && unzip -o ~/smr_updates69.zip && rm ~/smr_updates69.zip && git status --short"
    cd ~/smr && git add -A && git commit -m "v69: standard 7-Scenes/TUM protocol, VGGT-SLAM-style summary table" && git push

--------------------------------------------------------------------
## 0. Why
--------------------------------------------------------------------
The accepted protocol for feed-forward-backbone SLAM (MASt3R-SLAM ->
VGGT-SLAM -> ViSTA-SLAM / SLAM-Former / S-MUSt3R) is 7-Scenes seq-01 of
all seven scenes + the nine TUM fr1 sequences, ATE-RMSE after Sim(3),
averaged.  Stride-based systems in those tables use stride 5 (7-Scenes)
and 3 (TUM).  VGGT-SLAM (Maggio, Lim, Carlone 2025, arXiv 2505.12549)
is the direct prior work: VGGT submaps, retrieved loop frames APPENDED
TO THE SUBMAP PASS, factor graph on Sim(3)/SL(4).  Their uncalibrated
Sim(3) w=32 numbers: 7-Scenes avg 0.067 (chess 0.037, fire 0.026, heads
0.018, office 0.104, pumpkin 0.133, kitchen 0.061, stairs 0.093); TUM
avg 0.074 (360 0.123, desk 0.040, desk2 0.055, floor 0.254, plant
0.022, room 0.088, rpy 0.041, teddy 0.032, xyz 0.016); SL(4) TUM avg
0.053.  MASt3R-SLAM uncalibrated: 0.066 / 0.060.  Our Table 1 joins
that table; the multi-session runs are an ADDITIONAL stress test.

--------------------------------------------------------------------
## 1. What changed
--------------------------------------------------------------------
* scripts/prepare_standard_sets.sh: downloads the missing 7-Scenes
  scenes (heads, pumpkin, redkitchen, stairs) and TUM fr1 sequences
  (360, desk2, floor, plant, rpy, teddy, xyz), extracts GT (c2w).
* scripts/run_pilot_a_sweep.sh: strides 5 / 3 / 10 (7-Scenes single /
  TUM / multi-session); report names carry the stride.
* scripts/pilot_a_table.py: also prints the ATE-per-sequence summary in
  the VGGT-SLAM Table 1 layout (one block per backbone, avg column).
No library code changed; the test count is unchanged.

--------------------------------------------------------------------
## 2. Commands (after tonight's sweep finishes)
--------------------------------------------------------------------
    bash scripts/prepare_standard_sets.sh          # ~10 GB of downloads
    mv data/gt/7scenes_chess_seq01.npz /tmp/        # the old stride-10 copy is re-made at stride 5 by the sweep
    CUDA_VISIBLE_DEVICES=0 screen -S sweep0 -dm bash -c 'bash scripts/run_pilot_a_sweep.sh vggt pi3 dust3r 2>&1 | tee outputs/reports/sweep0.log'
    CUDA_VISIBLE_DEVICES=1 screen -S sweep1 -dm bash -c 'bash scripts/run_pilot_a_sweep.sh mast3r fast3r stream3r 2>&1 | tee outputs/reports/sweep1.log'
    CUDA_VISIBLE_DEVICES=2 screen -S sweep2 -dm bash -c 'bash scripts/run_pilot_a_sweep.sh streamvggt monst3r vggt_omega 2>&1 | tee outputs/reports/sweep2.log'
    # when all three are done:
    python scripts/pilot_a_table.py outputs/reports/pilotA_*.json --tex outputs/reports/table_pilotA.tex > outputs/reports/table_pilotA.md

Budget: 16 sequences x 9 backbones; VGGT/pi3-class ~40 min per backbone
over all 16, DUSt3R family ~2x.  Three GPUs in parallel: ~4-5 hours.

--------------------------------------------------------------------
## 3. Predictions
--------------------------------------------------------------------
* On 7-Scenes seq-01 (all seven), chained ~ smr ~ ceiling for VGGT and
  pi3: these sequences fit in 1-4 submaps (VGGT-SLAM's own appendix),
  so there is no drift to remove; our VGGT/pi3 numbers land near
  VGGT-SLAM Sim(3)'s per-scene numbers (chess ~0.04, office ~0.10,
  pumpkin ~0.13, stairs is the hard one).
* On TUM 360 / floor / room / teddy (long, loops) smr < chained; on
  desk / xyz / rpy / plant (short) chained ~ smr.  floor may fail the
  gate (planar; VGGT-SLAM diverges there too).
* The DUSt3R family chains worst and gains most in relative terms.
