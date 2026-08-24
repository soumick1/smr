# Pilot A — long-horizon global consistency. Run these on the server.

Everything below runs REAL backbones on REAL images against REAL
ground-truth poses. Nothing here is simulated. (v65 rebuilt the mechanism
and protocol; v66 added the consistency gate, local relaxation, the DINO
descriptor and multi-session sequences -- see UPDATE_NOTES_v66.md.)

--------------------------------------------------------------------
## Step 0 — sanity, no GPU (about 1 minute)
--------------------------------------------------------------------
    cd ~/smr && source .venv/bin/activate
    python -m pytest -q

Expected on the server: 147 passed, 23 skipped. The new tests pin the metrics to
hand-derivable cases, prove that `--sites 0` reproduces plain chaining to
machine precision, and show on a rendered two-lap world that anchoring
bounds the loop error where chaining compounds it. If anything fails,
stop; the table would be meaningless.

Optional, GPU-free harness demo (marked SIMULATED, never a result):

    python experiments/pilot_a.py --backbone synthetic --simulate-chunks \
        --chunk-noise 0.01 --chunk-distortion 0.1

--------------------------------------------------------------------
## Step 1 — data: 7-Scenes and TUM RGB-D (room walkthroughs with revisits)
--------------------------------------------------------------------
CO3D object orbits were the wrong regime for this pilot (every chunk is a
few degrees of arc; the models fail inside chunks). Indoor walkthroughs
are the regime where chunking is unavoidable and revisits exist, and
KV-Tracker (CVPR 2026) reports full-sequence ATE on exactly these
sequences.

7-Scenes (Microsoft; ~7 GB total, one scene at a time is fine):

    mkdir -p ~/7scenes && cd ~/7scenes
    for s in chess fire office; do
      wget -c http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/$s.zip
      unzip -q -o $s.zip && (cd $s && unzip -q -o seq-01.zip)
    done
    cd ~/smr

TUM RGB-D fr1 (a few hundred MB each):

    mkdir -p ~/tum && cd ~/tum
    for s in freiburg1_desk freiburg1_room freiburg1_xyz; do
      wget -c https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_$s.tgz
      tar xzf rgbd_dataset_$s.tgz
    done
    cd ~/smr

(If a URL has moved, the datasets are on the 7-Scenes and TUM RGB-D
project pages; the loaders only need the standard directory layouts.)

--------------------------------------------------------------------
## Step 2 — ground truth, with the convention MEASURED (5 minutes)
--------------------------------------------------------------------
    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes \
        --scene chess --seq 1 --diagnose-convention --backbone vggt

It runs VGGT on 10 frames spread over the sequence and scores every
reading of the stored pose (as-is / inverted, with each axis flip).
WHAT TO LOOK FOR: one convention wins by a mile (expected: `c2w`, AUC@30
well above the others). Then store with the winner:

    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes \
        --scene chess --seq 1 --convention c2w --out data/gt/7scenes_chess_seq01.npz
    python scripts/indoor_gt_poses.py tum --root ~/tum \
        --sequence rgbd_dataset_freiburg1_desk --diagnose-convention --backbone vggt
    python scripts/indoor_gt_poses.py tum --root ~/tum \
        --sequence rgbd_dataset_freiburg1_desk --convention c2w --out data/gt/tum_fr1_desk.npz

If a flipped variant wins instead, use it; send me the table.

--------------------------------------------------------------------
## Step 3 — probe first: is the backbone usable inside chunks? (10 min)
--------------------------------------------------------------------
    python experiments/pilot_a.py --gt data/gt/7scenes_chess_seq01.npz \
        --backbone vggt --keyframe-stride 10 --chunk 16 --overlap 8 --probe-only

Prints a REFERENCE pass (10 frames spread over the sequence — the
backbone's own wide-baseline ability on this scene) and every chunk
scored on its own. The GATE passes when the median chunk AUC@30 reaches
60% of the reference and the worst chunk clears 40. On 7-Scenes, VGGT
and pi3 score ~74–77 AUC@30 in their own papers, so expect the reference
in that range, not the 99 we saw on object orbits.

If the gate fails, sweep the geometry until it passes:

    for st in 5 10 20; do for c in 12 16 24; do
      python experiments/pilot_a.py --gt data/gt/7scenes_chess_seq01.npz \
          --backbone vggt --keyframe-stride $st --chunk $c --overlap $((c/2)) --probe-only
    done; done

Every pass is cached (outputs/cache/*.npy), so nothing is inferred twice.

--------------------------------------------------------------------
## Step 4 — the experiment, one backbone (20–40 minutes)
--------------------------------------------------------------------
    python experiments/pilot_a.py --gt data/gt/7scenes_chess_seq01.npz \
        --backbone vggt --keyframe-stride 10 --chunk 16 --overlap 8 --sites 2 \
        --rows chained,smr,smr_pgo,classical,smr_jump,plain --ceiling --verbose

Rows (all from the same cached passes where they share them):
  chained     sequential Sim(3), no memory — the baseline
  smr         memory-anchored streaming loop closure (the mechanism)
  smr_pgo     batch pose-graph solve on the SAME edges — the upper bound
  classical   descriptor-NN loop detection + batch PGO, no scaffold
  smr_jump    ablation: anchors, chunk snapped to them, nothing relaxed
  plain       ablation: plain descriptor index instead of the scaffold
Options: --correction relax|distribute|jump|none, --descriptor rgb|dino,
--remeasure (4-frame second-pass verification, ablation).
A `ceiling` row appears when the single pass covers every keyframe.
--ceiling also measures the largest single pass that fits (OOM bisection)
and its quality: the backbone's own ceiling and where chunking becomes
mandatory.

Columns: ATE (m); RPE at 1 keyframe and at 1 chunk; AUC@30 pooled /
within-chunk / cross-chunk; loop-closure rotation and translation error
over GT revisit pairs; max local-scale drift; closures accepted /
rejected by the consistency gate; frames per pass; seconds per frame. The `vs chained` lines are the headline ratios.

WHAT TO LOOK FOR
  * AUCin should be close across rows (it is the backbone's within-pass
    quality; a large drop in the smr rows means adding anchors changed the
    chunk's own geometry — report it, do not hide it).
  * The claim lives in loopR / loopT and AUCx: chained should be worst,
    smr and smr_pgo best. If smr ≈ chained with loops > 0, the anchors are
    being proposed but the placement is not helping — send me the events
    block of the JSON.
  * plain ≈ smr is EXPECTED (prediction recorded in advance): the address
    is not what buys the trajectory; loop closure is.
  * `--verbose` prints one line per chunk: how many sites, whether a loop
    closed, and the drift D it exposed.

--------------------------------------------------------------------
## Step 5 — every backbone, one process each; then the other sequences
--------------------------------------------------------------------
    for bb in vggt pi3 dust3r mast3r fast3r stream3r streamvggt monst3r vggt_omega; do
      python experiments/pilot_a.py --gt data/gt/7scenes_chess_seq01.npz \
          --backbone $bb --keyframe-stride 10 --chunk 16 --overlap 8 --sites 2 \
          --rows chained,smr,smr_pgo,classical --ceiling \
          --json outputs/reports/pilotA_chess_$bb.json
    done

One process per backbone is required, not stylistic: six of these repos
vendor forks of dust3r/croco and shadow each other in a shared process.
Start with vggt and pi3; if the mechanism does nothing on those two,
running the other seven changes only the clock. Then fire, office, and
TUM fr1_desk / fr1_room with the same command.

--------------------------------------------------------------------
## Step 6 — the length sweep (the drift curve)
--------------------------------------------------------------------
    for n in 40 80 160 320; do
      python experiments/pilot_a.py --gt data/gt/7scenes_chess_seq01.npz \
          --backbone vggt --keyframe-stride 5 --chunk 16 --overlap 8 --sites 2 \
          --max-frames $n --rows chained,smr,smr_pgo \
          --json outputs/reports/pilotA_chess_vggt_len$n.json
    done

Prediction: chained ATE and loop error grow with length; smr stays
bounded by roughly one pass of backbone error once the first loop closes.
If the ratio is flat, the mechanism is not doing its job, and we say so.

Paste the console output of Steps 3 and 4 back to me verbatim.

--------------------------------------------------------------------
## Step 7 — multi-session sequences (where single passes cannot fit)
--------------------------------------------------------------------
    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes --scene chess \
        --seq 1,2,3,4,5,6 --convention c2w --out data/gt/7scenes_chess_s0106.npz
    python experiments/pilot_a.py --gt data/gt/7scenes_chess_s0106.npz --backbone vggt \
        --keyframe-stride 10 --chunk 16 --overlap 8 --sites 2 \
        --rows chained,smr,smr_pgo --ceiling --verbose

Sessions of one 7-Scenes scene share a world frame, so this is a 6000-
frame trajectory with cross-session revisits and ground truth.  The
ceiling bisection reports the largest single pass that fits.
