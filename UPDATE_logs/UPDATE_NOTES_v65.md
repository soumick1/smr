# smr_updates65 — Pilot A rebuilt: the mechanism, the baselines, the metrics, the data

Apply exactly as before:

    scp -P 22 /mnt/c/MyFiles/smr_updates/smr_updates65.zip soumick@10.97.144.63:~/
    ssh -p 22 soumick@10.97.144.63 "cd ~/smr && unzip -o ~/smr_updates65.zip && rm ~/smr_updates65.zip && git status --short"
    # then on the server
    cd ~/smr && source .venv/bin/activate && python -m pytest -q      # expect 154 passed, 10 skipped
    git add -A && git commit -m "v65: memory-anchored stitching, PGO baseline, long-horizon metrics, 7-Scenes/TUM loaders" && git push

--------------------------------------------------------------------
## 0. The error, stated plainly
--------------------------------------------------------------------
The v64 `stitch_smr` could never beat the chained baseline, on any dataset.
`place_pose` followed by `state()` is ignite-and-settle at a CONTINUOUS
phase: there is no lattice and no g -> h -> g clean-up in that path, and the
memory module is never touched. So `decoded[k] ~= glob[k] + jitter` and the
two rows tie by construction. The docstring's claim that poses were "snapped
to the lattice" was my error. Four failed CO3D runs were consistent with
this and I attributed them to chunk geometry alone; that part was also
wrong. The regime diagnosis (chunks of a few degrees of arc are degenerate
for these models) still stands and is why the pilot moves to indoor
walkthroughs.

The corrected physics: attractor error-correction is coherent-blind (it
removes off-manifold components); pose error is ON the manifold. The only
thing that can bound drift is a non-sequential constraint, i.e. loop
closure, and the only thing in SMR that can supply one is the memory:
earlier views of the same place, proposed by appearance and by place,
verified by geometry, and used as anchors. That is what v65 builds.

--------------------------------------------------------------------
## 1. What changed
--------------------------------------------------------------------
New package `src/smr/stitch/` (all numpy/scipy, fully tested here):

* `anchored.py`   AnchoredStitcher — per chunk: propose anchor SITES (an old
                  view + its stored temporal neighbour, so every site gives
                  a verifiable relative pose and a scale) from the new
                  frames' descriptors and from the chain's predicted place;
                  ONE backbone pass on chunk ∪ anchors; verify each site
                  (pass relative pose vs stored relative pose: rot < 10 deg,
                  direction < 25 deg); S_A = robust Sim(3) onto the overlap
                  (the chained placement), S_B = robust Sim(3) onto verified
                  old anchors (the loop placement). Chunk placed by S_B when
                  it exists; the exposed drift D = S_B ∘ inv(S_A) is
                  distributed over the un-anchored stretch since the last
                  closure (`correction="distribute"`), or not (`"jump"`).
                  `--sites 0` reproduces plain chaining to machine precision
                  (tested), so the mechanism IS the anchors. Every pass and
                  every accepted edge is recorded for the batch solver.
* `posegraph.py`  batch Sim(3) pose-graph optimisation (scipy
                  least_squares, Huber f_scale 0.5, equal edge weights) over
                  the IDENTICAL edges the streaming stitcher recorded: the
                  batch upper bound. Equal weights matter: weighting by
                  sqrt(#correspondences) favours the 8-frame overlap edges
                  over 4-frame loop edges and, measured in simulation, left
                  ~2x more loop error — a handicapped upper bound would
                  flatter the streaming row, so it is not used.
* `memory_index.py` two place indices behind one interface: ScaffoldIndex
                  (grid-code address h from the module bumps, k-WTA, RLS
                  s<->h binding both ways, proposal by cueing s -> h and
                  scoring stored addresses by overlap) and DescriptorIndex
                  (same descriptors, plain cosine NN, no address). The
                  pilot encodes addresses analytically (template bumps at
                  the pose's phases); the real bump dynamics give the same
                  address (cos 0.95–0.98 at the same pose vs 0.40 at a pose
                  0.6 units away — tested). `scaffold_decode` (continuity
                  unwrap) moved here from the old pilot.
* `passes.py`     PassCache (memoises passes by ordered frame tuple,
                  persists to disk, so reruns never re-infer and rows that
                  share passes see identical outputs), BackboneRunner (real
                  backbones), SimulatedRunner (GT + per-pass similarity +
                  within-pass GEOMETRIC distortion ramp + jitter),
                  image descriptors from raw files (PIL, 160 px).
* `probe.py`      per-chunk probe + a REFERENCE pass (n frames spread over
                  the sequence) + gate: median chunk AUC >= 60% of the
                  reference and worst >= 40.
* `sim3.py`       compose/inverse/interpolate/fit_poses (orientation-aware,
                  2 poses suffice) / fit_poses_robust (iterative outlier
                  rejection — one false anchor cannot move a chunk).
* `chunks.py`, `simulate.py` (rendered two-lap room with revisits, for
                  GPU-free validation).

Changed:
* `src/smr/backbones/base.py` — the view descriptor is now a module
  function `rgb_descriptor(img, dim)` so it can be computed from a raw image
  BEFORE any backbone pass (anchor proposal needs it) with the identical
  function that binds it. `BackboneOutput.descriptor` unchanged in value.
* `src/smr/eval/trajectory.py` — added `find_revisit_pairs`,
  `loop_closure_error` (relative pose across GT revisit pairs; rotation
  alignment-free, translation needs only the global scale), `auc_split`
  (within-pass vs cross-pass AUC@30 — the pass-through invariant made
  visible), `scale_drift`, `rpe_at_distance` (RPE per metre of GT path),
  `summarise_long`. Nothing existing was modified.
* `experiments/pilot_a.py` — rewritten around the package. Rows: chained,
  smr, smr_pgo, classical (descriptor-NN loops + PGO), smr_jump, plain.
  `--probe-only`, `--ceiling` (largest single pass that fits, OOM bisection
  over frames spread across the sequence), `--cache`, `--max-frames`,
  `--rpe-dist`, `--revisit-*`, `--verbose` (one line per chunk: sites,
  loop, exposed drift). Reports carry probe, gate, ceiling, rows, events.
* `scripts/indoor_gt_poses.py` (new) — 7-Scenes (`frame-*.pose.txt`, 4x4
  c2w, invalid frames dropped and counted) and TUM RGB-D (`rgb.txt` <->
  `groundtruth.txt` nearest-timestamp association within 0.02 s,
  quaternion xyzw) into the pilot's npz schema, with
  `--diagnose-convention --backbone X`: scores as-is / inverted x four
  axis flips against a real backbone pass on frames spread over the
  sequence. The convention is measured, not assumed.
* `RUN_PILOT_A.md` — rewritten for the new flow.
* `tests/test_trajectory.py` — the four tests that loaded the old pilot
  module now target the package; the source-inspection test became a
  behavioural one (collinear overlaps chain exactly).

New tests (26): `test_sim3.py` (7), `test_trajectory_long.py` (6),
`test_stitch.py` (9, ~13 s, rendered world), `test_indoor_loaders.py` (4).
Full suite here: 154 passed, 10 skipped (was 128 / 10).

--------------------------------------------------------------------
## 2. Protocol decisions (so the table matches the literature)
--------------------------------------------------------------------
* Datasets: 7-Scenes seq-01 (chess, fire, office first) and TUM RGB-D fr1
  (desk, room). Room walkthroughs are the regime where chunking is
  unavoidable and revisits exist. KV-Tracker (Davison lab, CVPR 2026,
  arXiv 2512.22581) reports full-sequence ATE-RMSE after Sim(3) Umeyama
  on exactly these; their Table rows (7-Scenes avg: Point3R 0.439, CUT3R
  0.205, TTT3R 0.143, DA3 0.118, KV-Tracker 0.080 m; TUM fr1: CUT3R 0.272,
  TTT3R 0.132, DPVO 0.095, KV-Tracker 0.108 m) become external rows once
  our numbers exist. CUT3R/TTT3R were run there with state reset every
  100 frames — that IS chunking, so the comparison is fair by
  construction.
* Metric: ATE-RMSE after Sim(3) alignment (our `ate_rmse`), plus the
  long-horizon set above. AUC@30 as in VGGT (`auc_at` matches VGGT's
  definition; do the 5-minute diff against their `calculate_auc_np` on the
  server when convenient).
* Gate: relative to a reference pass on the same scene, not an absolute
  99. VGGT-Ω Table 1 gives 7-Scenes AUC@30 VGGT 74.4 / pi3 77.0; expect
  the reference there.
* Keyframes: stride 10 by default (consecutive video frames are a
  degenerate baseline for these models); sweep 5/10/20 with the probe if
  the gate fails.

--------------------------------------------------------------------
## 3. Tested here vs first execution on your server
--------------------------------------------------------------------
Tested here (no torch in this sandbox): every numpy/scipy path — metrics,
Sim(3) helpers, both indices, the streaming stitcher, the batch solver,
the probe/gate, the pass cache, both loaders (synthetic fixtures), the
synthetic pilot end to end.

First execution on your server: `BackboneRunner` (real inference through
`get_backbone(name, device).infer(paths)` — the same call v64 made),
`image_descriptors` on real PNGs (PIL path), `--ceiling` OOM catching,
`--diagnose-convention`. These are thin and mirror v64 code that ran, but
they have not run here. If any of them throws, paste the traceback.

SIMULATED validation (harness only, never a result): rendered two-lap
room, distortion 0.1, 5 seeds — chained ATE 0.330 / loop-rot 13.3 deg /
cross-pass AUC 39.5 -> smr 0.125 / 2.45 / 68.1 -> smr_pgo ~0.10.
plain-index 0.125 / 3.15 / 69.7 (the tie predicted below).

--------------------------------------------------------------------
## 4. Commands, in order
--------------------------------------------------------------------
    # data (one scene at a time is fine)
    mkdir -p ~/7scenes && cd ~/7scenes
    for s in chess fire office; do
      wget -c http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/$s.zip
      unzip -q -o $s.zip && (cd $s && unzip -q -o seq-01.zip)
    done
    mkdir -p ~/tum && cd ~/tum
    for s in freiburg1_desk freiburg1_room; do
      wget -c https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_$s.tgz && tar xzf rgbd_dataset_$s.tgz
    done
    cd ~/smr && source .venv/bin/activate

    # ground truth, convention MEASURED against VGGT, then stored
    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes --scene chess --seq 1 \
        --diagnose-convention --backbone vggt
    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes --scene chess --seq 1 \
        --convention c2w --out data/gt/7scenes_chess_seq01.npz          # use the winner
    python scripts/indoor_gt_poses.py tum --root ~/tum --sequence rgbd_dataset_freiburg1_desk \
        --diagnose-convention --backbone vggt
    python scripts/indoor_gt_poses.py tum --root ~/tum --sequence rgbd_dataset_freiburg1_desk \
        --convention c2w --out data/gt/tum_fr1_desk.npz

    # probe: is VGGT usable inside chunks on this scene?
    python experiments/pilot_a.py --gt data/gt/7scenes_chess_seq01.npz --backbone vggt \
        --keyframe-stride 10 --chunk 16 --overlap 8 --probe-only
    # if the gate fails, sweep: --keyframe-stride {5,10,20} x --chunk {12,16,24} (overlap = chunk/2)

    # the experiment, vggt then pi3 (one process each; screen recommended)
    for bb in vggt pi3; do
      python experiments/pilot_a.py --gt data/gt/7scenes_chess_seq01.npz --backbone $bb \
          --keyframe-stride 10 --chunk 16 --overlap 8 --sites 2 \
          --rows chained,smr,smr_pgo,classical,smr_jump,plain --ceiling --verbose \
          --json outputs/reports/pilotA_chess_$bb.json 2>&1 | tee outputs/reports/pilotA_chess_$bb.log
    done

    # then the remaining seven, the other scenes, and TUM, with --rows chained,smr,smr_pgo,classical
    # and the length sweep (RUN_PILOT_A.md Step 6)

Paste the console output of the probe and of the vggt / pi3 runs back
verbatim (the .log files).

--------------------------------------------------------------------
## 5. Predictions recorded before the data (so we cannot move them after)
--------------------------------------------------------------------
1. AUCin (within-pass) will be nearly identical across rows that share
   passes and close between chained and smr rows; a large drop in the smr
   rows would mean adding anchors changed the chunk's own geometry.
2. loopR / loopT and AUCx (cross-pass) are where the claim lives:
   chained worst, smr_pgo best, smr close to smr_pgo. The length sweep
   shows chained growing and smr bounded once the first loop closes.
3. `plain` (descriptor NN) will close the same loops and land within
   noise of `smr` on every trajectory column. The scaffold address is
   NOT what buys the trajectory result and the paper will say so; the
   address earns its keep in rendering/imagination/novelty (Pilots B–D).
4. `classical` (same detector, batch PGO) ~ `smr_pgo`; the only difference
   is the index. If either beats the other consistently it is measurable
   and reportable, not something to argue about.
5. pi3 rows will show smaller within-pass distortion than VGGT (pi3 drift
   was flat at 1–4 deg on LLFF; VGGT rose to ~29 deg), so the chained
   baseline will be less bad on pi3 and the smr gain smaller in absolute
   terms.
6. Streaming cost: one extra pass of (chunk + 2*sites) frames per chunk;
   memory writes are O(N_h^2) per frame and negligible next to inference.

If (2) fails with loops > 0, the anchors are proposed but the placement
does not help — send the `events` block; if loops == 0, proposal or
verification is too strict for real revisits — send the per-chunk
`--verbose` lines (each shows the sites, their descriptor cosine and the
rotation/direction residuals that rejected them).

--------------------------------------------------------------------
## 6. Deferred to v66 (after these numbers exist)
--------------------------------------------------------------------
* multi-session relocalisation (bind seq A, relocalise seq B; recall @
  5 cm / 5 deg) — the capability chunking baselines do not have;
* scaling sweep vs N with OOM marks (memory, s/frame) for all rows;
* KV-Tracker / CUT3R / TTT3R comparison rows in the report generator;
* an independent classical loop detector (not our descriptor) as a
  stronger external baseline;
* CO3D short-sequence parity table (the pass-through invariant, shown);
* paper: demote biology to motivation, promote the plug-in/consistency
  framing; Table 1 = this pilot.
