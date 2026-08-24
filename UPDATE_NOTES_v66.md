# smr_updates66 — consistency-gated loop closure, local relaxation, multi-session data

    scp -P 22 /mnt/c/MyFiles/smr_updates/smr_updates66.zip soumick@10.97.144.63:~/
    ssh -p 22 soumick@10.97.144.63 "cd ~/smr && unzip -o ~/smr_updates66.zip && rm ~/smr_updates66.zip && git status --short"
    cd ~/smr && source .venv/bin/activate && python -m pytest -q     # expect 147 passed, 23 skipped
    git add -A && git commit -m "v66: consistency-gated closures, local relaxation, DINO descriptor, 7-Scenes session concat" && git push

(pyproject.toml and tests/test_backbones.py in this zip are byte-identical
to what you already applied by hand; unzip -o will report no change.)

--------------------------------------------------------------------
## 0. What the first real runs said
--------------------------------------------------------------------
7-Scenes chess seq-01, 100 keyframes: chained ATE 0.041 m = single-pass
ceiling 0.039 m.  No drift to remove; every stitcher sits on the
backbone's own global-consistency floor.  Streaming smr LOST (0.062)
because every closure set the chunk's scale from a 2-frame site a few
centimetres wide (D_logs swings of 10-30%).  Batch smr_pgo won slightly
(0.039).  The events showed both.

TUM fr1_room, VGGT, 273 keyframes / 34 chunks: chained 0.120 m vs ceiling
0.0896 m (+34%), loop error 5.7 deg / 19.5 cm -> the regime where a
stitcher has something to do.  smr was destroyed (0.918): 22 closures
accepted, >= 13 grossly false (D_rot 21-166 deg, D_t up to 4.9 units,
scale x24).  Cause, from the events: the pooled-RGB descriptor cannot
separate a revisit (cosine 0.33-0.58) from any other view of the room
(up to 0.77); my site check compared the two site frames' relative pose
with memory, which any adjacent pair anywhere satisfies -- it verified a
site against itself, never against the chunk.  True closures had D_rot
<= 4.4 deg; false ones >= 21 deg.  That separation is what v66 uses.
Both failures were mine; the harness, cache, probe, loaders and
convention diagnosis worked as designed.

--------------------------------------------------------------------
## 1. What changed
--------------------------------------------------------------------
`src/smr/stitch/anchored.py` (rewritten):
* CONSISTENCY GATE: a closure is accepted only if the discrepancy
  D = S_B o inv(S_A) between the loop placement and the chained placement,
  evaluated AT THE CHUNK (rotation; position in units of the chunk's
  spread; |log scale| when scale is measurable), lies inside a drift
  budget that grows with the number of un-anchored chunks since the last
  closure: rot <= min(45, 10 + 3 n), pos <= 1.0 + 0.5 n spreads,
  |log s| <= 0.5.  Rejected closures are counted (`rej` column) and
  logged with the reason and the budget.  In simulation with a
  co-visibility model (a frame that shares no view direction with the
  chunk gets an arbitrary pose per pass, as a real model does), the gate
  plus the robust fit reject 35/35 injected look-alike sites with the
  extent check disabled, and the poisoned run stays within noise of the
  clean one (tested).
* EXTENT CHECK: a site the backbone placed farther than 3 chunk spreads
  from the chunk's centroid in pass coordinates is rejected.
* SPACED PARTNERS: a site's partner is 3 keyframes away (not adjacent),
  so the site's internal relative pose carries some information and its
  baseline can measure scale.
* SCALE GUARD: if the verified anchors' baseline is under 25% of the
  chunk's spread, the loop placement takes its scale from the overlap
  fit and the loop edge's scale component gets zero weight in the batch
  solve (`fit_poses_fixed_scale`).
* correction="relax" (new default): an accepted closure triggers a local
  pose-graph solve over the chunks since the last closure with earlier
  chunks fixed -- the amortised, streaming form of the batch back-end,
  cost O(stretch).  "distribute", "jump", "none" remain as ablations
  (`--correction`).  Relax agrees with the batch solve on a single loop
  (tested).
* `--remeasure` (off by default): re-run the backbone on a 4-frame pass
  (two best-matching chunk frames + the site) and require the
  chunk->anchor relative pose to agree with the big pass.  Implemented
  and tested; in simulation it rejects true closures (5.0 -> 2.8 per
  run) while the gate alone already catches garbage anchors, so it is an
  ablation for the real-data failure mode the simulator cannot produce.
`src/smr/stitch/posegraph.py`: `solve_nodes(nodes, edges, free)` with
  fixed nodes and per-edge scale weights; batch `solve` built on it.
`src/smr/stitch/passes.py`: `dino_descriptors` (DINOv2 ViT-S/14 CLS,
  384-d, via torch.hub -- FIRST RUN ON THE SERVER, downloads weights);
  SimulatedRunner `covis_deg` co-visibility model.
`src/smr/stitch/memory_index.py`: `desc_dim` (the RLS memory binds any
  descriptor width).
`experiments/pilot_a.py`: `--correction`, `--remeasure`, `--descriptor
  rgb|dino`; the ceiling becomes a table row when it covers every
  keyframe; AUCin/AUCx are now split by CHUNK WINDOW for every row (the
  by-pass split is kept in the JSON as auc_within_pass/auc_cross_pass);
  `rej` column.
`scripts/indoor_gt_poses.py`: `--seq 1,2,3` concatenates sessions of one
  7-Scenes scene (they share a world frame, so cross-session revisits
  carry GT; frame ids offset by 100000 x seq).
Tests: 160 passed, 10 skipped here (+6: gate, budget, relax = batch,
  fixed-scale fit, local solver, session concat).  On your server expect
  147 passed, 23 skipped.

--------------------------------------------------------------------
## 2. Commands
--------------------------------------------------------------------
    cd ~/smr && source .venv/bin/activate

    # (a) TUM room again -- the regime with drift.  Chunk passes are cached;
    #     anchored passes are new (partners changed).  ~15 min.
    python experiments/pilot_a.py --gt data/gt/tum_fr1_room.npz --backbone vggt \
        --keyframe-stride 5 --chunk 16 --overlap 8 --sites 2 \
        --rows chained,smr,smr_pgo,classical,smr_jump,plain --ceiling --verbose \
        --json outputs/reports/pilotA_room_vggt_s5_v66.json 2>&1 | tee outputs/reports/pilotA_room_vggt_s5_v66.log

    # (b) same with the DINO descriptor (first run downloads dinov2_vits14)
    python experiments/pilot_a.py --gt data/gt/tum_fr1_room.npz --backbone vggt \
        --keyframe-stride 5 --chunk 16 --overlap 8 --sites 2 --descriptor dino \
        --rows chained,smr,smr_pgo --verbose \
        --json outputs/reports/pilotA_room_vggt_s5_dino.json 2>&1 | tee outputs/reports/pilotA_room_vggt_s5_dino.log

    # (c) the multi-session regime: chess seq-01..06 (6000 frames, 600 keyframes)
    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes --scene chess --seq 1,2,3,4,5,6 \
        --convention c2w --out data/gt/7scenes_chess_s0106.npz
    python experiments/pilot_a.py --gt data/gt/7scenes_chess_s0106.npz --backbone vggt \
        --keyframe-stride 10 --chunk 16 --overlap 8 --probe-only
    python experiments/pilot_a.py --gt data/gt/7scenes_chess_s0106.npz --backbone vggt \
        --keyframe-stride 10 --chunk 16 --overlap 8 --sites 2 \
        --rows chained,smr,smr_pgo --ceiling --verbose \
        --json outputs/reports/pilotA_chess6_vggt.json 2>&1 | tee outputs/reports/pilotA_chess6_vggt.log

Paste the three logs.  (c)'s ceiling will OOM-bisect; the largest fitting
pass is the backbone's own limit and belongs in the paper.

--------------------------------------------------------------------
## 3. Predictions, recorded now
--------------------------------------------------------------------
1. Room (a): rej >= 8 of the previous 22; smr ATE between chained (0.120)
   and the ceiling (0.090); loop error below chained's 5.7 deg; AUCin
   within ~2 of chained's.  smr_pgo <= smr.  If smr is still above
   chained, the events will show which accepted closure did it and (b)
   or --remeasure is the next lever.
2. (b): fewer proposals, fewer rejections, same or better trajectory
   numbers than (a).  If DINO alone fixes it, the paper's descriptor is
   DINO and the RGB one is retired.
3. (c): chained drift grows over 75 chunks and the single pass does not
   fit at 600 keyframes; smr stays bounded once sessions revisit.  This
   is the regime for Table 1.
