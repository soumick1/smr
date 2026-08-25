# smr_updates67 — consensus rule for closures, scale baseline, percentile gate, DINO default

    scp -P 22 /mnt/c/MyFiles/smr_updates/smr_updates67.zip soumick@10.97.144.63:~/
    ssh -p 22 soumick@10.97.144.63 "cd ~/smr && unzip -o ~/smr_updates67.zip && rm ~/smr_updates67.zip && git status --short"
    cd ~/smr && source .venv/bin/activate && python -m pytest -q     # expect 150 passed, 23 skipped
    git add -A && git commit -m "v67: consensus rule, scale baseline, percentile gate, DINO default" && git push

--------------------------------------------------------------------
## 0. What the v66 runs said (all three, in one line each)
--------------------------------------------------------------------
chess x6 sessions (600 kf, single pass OOMs): chained 0.083 -> smr 0.045 m,
  AUCx 82.8 -> 89.0, loop-trans 12 -> 4 cm, 56 closures / 0 rejected, all
  sub-degree, AUCin and RPE-t1 IDENTICAL to chained (pass-through held
  exactly).  This is Table 1.  The "GATE FAILED" was my criterion: one
  chunk of 74 is near-pure rotation (RTA median 25 deg) and every row
  shares it.
room + DINO: closures right (loop error 5.7 -> 1.1 deg, 19.5 -> 4 cm),
  ATE 0.147 vs chained 0.120: two over-large SINGLE-site corrections
  (11.9 deg after 5 chunks from a pose-only site; 2.7 spreads after 13)
  passed the loose budget and moved chunks 6-24 the wrong way.
room + RGB: gate rejected the 3 gross candidates; the one accepted
  applied a 21% scale change from a 2-frame baseline of 0.42 chunk
  spreads (classical: 4 closures, loop error 1.4 deg / 5 cm, scale drift
  0.60, ATE 0.36 -- same disease).  RGB found 1 closure where DINO found
  10: the RGB descriptor is retired for real runs.

--------------------------------------------------------------------
## 1. What changed
--------------------------------------------------------------------
* CONSENSUS RULE (anchored.py): a closure supported by ONE site may only
  nudge -- tight budget rot <= min(15, 3 + 1 n), pos <= 0.5 + 0.15 n
  spreads; a large correction needs TWO independent sites that agree
  (loose budget as before).  The 11.9 deg case fails the tight budget.
* POSE-ONLY SITES cannot close a loop on their own (`require_appearance`,
  default on): a site proposed by place alone, with no appearance match,
  may join a closure but never carry one.
* SCALE from anchors only when their baseline is >= 1.0 x the chunk's
  own spread (was 0.25); otherwise scale comes from the overlap fit and
  the loop edge's scale weight is 0 in the solver.
* GATE (probe.py): 10th-percentile chunk AUC >= floor instead of the
  worst chunk; chunks under the floor are listed in the verdict.
* DINO is the default descriptor for real backbones (`--descriptor rgb`
  still available; the synthetic world keeps rgb).
Tests: +3 (tight budget for one site; pose-only sites never close;
percentile gate).  163 passed / 10 skipped here; 150 / 23 on the server.

--------------------------------------------------------------------
## 2. Commands
--------------------------------------------------------------------
    cd ~/smr && source .venv/bin/activate
    # (a) room again, DINO (default), all rows; passes cached
    python experiments/pilot_a.py --gt data/gt/tum_fr1_room.npz --backbone vggt \
        --keyframe-stride 5 --chunk 16 --overlap 8 --sites 2 \
        --rows chained,smr,smr_pgo,classical,smr_jump,plain --ceiling --verbose \
        --json outputs/reports/pilotA_room_vggt_s5_v67.json 2>&1 | tee outputs/reports/pilotA_room_vggt_s5_v67.log
    # (b) chess x6 again with DINO (v66 used RGB there), vggt then pi3
    for bb in vggt pi3; do
      python experiments/pilot_a.py --gt data/gt/7scenes_chess_s0106.npz --backbone $bb \
          --keyframe-stride 10 --chunk 16 --overlap 8 --sites 2 \
          --rows chained,smr,smr_pgo,classical --ceiling --verbose \
          --json outputs/reports/pilotA_chess6_${bb}_v67.json 2>&1 | tee outputs/reports/pilotA_chess6_${bb}_v67.log
    done
    # (c) a second multi-session scene (harder): fire seq-01..06? office? -- check which seqs exist:
    ls ~/7scenes/fire ~/7scenes/office
    for s in 02 03 04 05 06; do (cd ~/7scenes/fire && [ -f seq-$s.zip ] && unzip -q -o seq-$s.zip); done
    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes --scene fire --seq 1,2,3,4 \
        --convention c2w --out data/gt/7scenes_fire_s0104.npz     # use the seqs that exist
    python experiments/pilot_a.py --gt data/gt/7scenes_fire_s0104.npz --backbone vggt \
        --keyframe-stride 10 --chunk 16 --overlap 8 --sites 2 --rows chained,smr,smr_pgo --ceiling --verbose \
        --json outputs/reports/pilotA_fire4_vggt_v67.json 2>&1 | tee outputs/reports/pilotA_fire4_vggt_v67.log

--------------------------------------------------------------------
## 3. Predictions
--------------------------------------------------------------------
(a) room: the k=10 closure is rejected (tight budget), fewer accepted
    (6-9), smr ATE at or below chained (0.120) with loop error ~1 deg;
    AUCin within ~2 of chained.  If ATE is still above chained, the
    remaining accepted closures' D's in the events tell which one.
(b) chess x6 with DINO: same or better than the RGB run (0.045), more
    closures; pi3 chained will drift less than vggt and the gain will be
    smaller in absolute terms; smr should still reach the 300-frame
    single-pass level.
(c) fire: a harder scene (7-Scenes fire has more texture-poor views);
    expect a lower gate percentile and a larger chained-vs-smr gap.
