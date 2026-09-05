# smr_updates68 — junction blending, baseline-weighted loop scale, sweep + table + drift-curve tooling

    scp -P 22 /mnt/c/MyFiles/smr_updates/smr_updates68.zip soumick@10.97.144.63:~/
    ssh -p 22 soumick@10.97.144.63 "cd ~/smr && unzip -o ~/smr_updates68.zip && rm ~/smr_updates68.zip && git status --short"
    cd ~/smr && source .venv/bin/activate && python -m pytest -q     # expect 151 passed, 23 skipped
    git add -A && git commit -m "v68: junction blending, loop-edge scale weight, sweep/table/length tooling" && git push

--------------------------------------------------------------------
## 0. What the v67 runs said
--------------------------------------------------------------------
room (VGGT, DINO): smr 0.092 m vs chained 0.120 vs ceiling 0.090; loop
  error 5.7 deg / 19.5 cm -> 1.0 deg / 3.9 cm; 10 closures, 1 rejected;
  the 11.9 deg single-site closure is gone (consensus rule).  Cost: AUCin
  80.8 -> 75.8 and RPE-t1 0.019 -> 0.025 -- the junction step left by
  relaxing whole chunks.  classical 0.139: eight two-frame loop edges
  with noisy scale in the batch solve (scl 0.46).
chess x6 (DINO): VGGT 0.083 -> 0.049 (ceiling 0.048 on a 300-frame
  subset); pi3 0.068 -> 0.046 (ceiling 0.045 on 75 frames); 61 / 59
  closures, 0 rejected; AUCin identical to chained; AUCx 82.8 -> 88.0.
  Every prediction in UPDATE_NOTES_v67 held.  The paper has these
  numbers (Overleaf update shipped alongside this zip).

--------------------------------------------------------------------
## 1. What changed
--------------------------------------------------------------------
* JUNCTION BLENDING (anchored.py, `smooth_junctions=True`): after an
  accepted closure the overlap frames are blended between their owner's
  placement and chunk k's, first frame with the owner, last almost with
  k.  Simulation (5 seeds): AUCin 85.4 -> 86.7 (chained 86.5), RPE-t1
  0.033 -> 0.027 (chained 0.031), ATE 0.072 -> 0.064.  Off when no
  closure was accepted, so the chained reduction is untouched (tested).
* LOOP-EDGE SCALE WEIGHT proportional to baseline: w_scale = min(1,
  baseline/2 spreads) instead of 1; the batch solve trusts a two-frame
  scale only as far as its baseline earns it.
* scripts/pilot_a_table.py: every report -> one markdown + LaTeX table
  (skips SIMULATED reports; ceiling row with frame count).
* scripts/run_pilot_a_sweep.sh: all backbones x all data/gt/*.npz, one
  process each, skips existing reports, ends with the table.
* scripts/run_length_sweep.sh + scripts/plot_length_sweep.py: the
  drift-versus-length curve (100..600 keyframes) with the single-pass
  points where they fit -> outputs/figures/length_sweep_*.pdf.
Tests: +1 (junction blending restores within-chunk accuracy).  164 /
10 here; 151 / 23 on the server.

--------------------------------------------------------------------
## 2. Commands
--------------------------------------------------------------------
    # (a) room + chess x6 again with the junction blend (passes cached: minutes)
    python experiments/pilot_a.py --gt data/gt/tum_fr1_room.npz --backbone vggt --keyframe-stride 5 \
        --chunk 16 --overlap 8 --sites 2 --rows chained,smr,smr_pgo,classical --ceiling \
        --json outputs/reports/pilotA_tum_fr1_room_vggt.json 2>&1 | tee outputs/reports/pilotA_tum_fr1_room_vggt.log
    for bb in vggt pi3; do
      python experiments/pilot_a.py --gt data/gt/7scenes_chess_s0106.npz --backbone $bb --keyframe-stride 10 \
          --chunk 16 --overlap 8 --sites 2 --rows chained,smr,smr_pgo,classical --ceiling \
          --json outputs/reports/pilotA_7scenes_chess_s0106_$bb.json 2>&1 | tee outputs/reports/pilotA_7scenes_chess_s0106_$bb.log
    done
    # (b) more scenes: fire and office sessions (unzip what exists), then TUM desk
    for sc in fire office; do for s in 02 03 04 05 06; do (cd ~/7scenes/$sc && [ -f seq-$s.zip ] && unzip -q -o seq-$s.zip); done; ls ~/7scenes/$sc; done
    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes --scene fire --seq 1,2,3,4 --convention c2w --out data/gt/7scenes_fire_s0104.npz
    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes --scene office --seq 1,2,3,4,5,6 --convention c2w --out data/gt/7scenes_office_s0106.npz
    python scripts/indoor_gt_poses.py tum --root ~/tum --sequence rgbd_dataset_freiburg1_desk --convention c2w --out data/gt/tum_fr1_desk.npz
    # (c) THE SWEEP: all nine backbones x all prepared sequences (hours; screen)
    bash scripts/run_pilot_a_sweep.sh
    # (d) the drift curve for the paper figure
    bash scripts/run_length_sweep.sh data/gt/7scenes_chess_s0106.npz vggt
    bash scripts/run_length_sweep.sh data/gt/7scenes_chess_s0106.npz pi3

Send back outputs/reports/table_pilotA.tex, the length-sweep PNGs, and
any run whose gate failed.

--------------------------------------------------------------------
## 3. Predictions
--------------------------------------------------------------------
(a) room: AUCin back within ~2 of chained (80.8), RPE-t1 back near
    0.020, ATE <= 0.092; chess numbers unchanged within 0.002.
(c) every backbone shows chained > smr on chess x6; the dust3r-family
    models (dust3r, mast3r, fast3r, monst3r) chain worse than VGGT/pi3
    because their per-pass AUC is lower (their reference-view coupling);
    stream3r / streamvggt / vggt_omega behave like VGGT.  If a backbone's
    gate fails, its chunk geometry (stride) needs the probe sweep first.
(d) chained ATE grows with N; smr flat once closures start (~100
    keyframes on chess); the single-pass points stop at 300 (VGGT) and
    75 (pi3).
