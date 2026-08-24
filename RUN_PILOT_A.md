# Pilot A — the actual evaluation. Run these on the server.

Everything below runs REAL backbones on REAL images against REAL
ground-truth poses. Nothing here is simulated.

--------------------------------------------------------------------
## Step 0 — sanity: 19 metric tests, no GPU needed (30 seconds)
--------------------------------------------------------------------
    cd ~/smr && source .venv/bin/activate
    python -m pytest tests/test_trajectory.py -v

These check the metrics against cases with known answers (a
similarity-transformed trajectory must score ATE exactly 0; an injected
5-degree per-step error must read back as 5 degrees; AUC matched to a hand
computation). If any fail, stop — the table would be meaningless.

--------------------------------------------------------------------
## Step 1 — ground truth (this is the part that was missing)
--------------------------------------------------------------------
    python scripts/co3d_gt_poses.py --root ~/co3d_data --list

Pick a category with a long sequence. Then extract AND verify the pose
convention against a real backbone:

    python scripts/co3d_gt_poses.py --root ~/co3d_data --category hydrant \
        --out data/gt/hydrant.npz --validate --backbone vggt

WHAT TO LOOK FOR: it prints AUC@30 for VGGT against the extracted GT.
  * AUC > 60  -> convention confirmed, proceed.
  * AUC near 0 with a sane ATE -> the camera-axis flip is wrong; send me
    the line and I will fix FLIP in that script.
This is a real check, not a formality: a wrong flip leaves camera CENTRES
correct (so ATE looks fine) while every orientation is off by 180 degrees.

NOTE: ~/co3d_full_* (the cycled categories) have NO annotations — an early
version of co3d_cycle.sh deleted *.jgz while pruning. That was my error.
Use ~/co3d_data (the download subset), which still has them.

--------------------------------------------------------------------
## Step 2 — the experiment, one backbone (about 10-20 minutes)
--------------------------------------------------------------------
    python experiments/pilot_a.py --gt data/gt/hydrant.npz \
        --backbone vggt --chunk 16 --overlap 4

Two rows come out: "backbone alone" and "+ SMR", from the SAME cached
backbone outputs, so the only difference between them is the stitching
mechanism. Columns: ATE, RPE-trans, RPE-rot, AUC@30, revisit drift, peak
memory, seconds per frame.

--------------------------------------------------------------------
## Step 3 — all nine backbones, one process each
--------------------------------------------------------------------
    for bb in vggt pi3 dust3r mast3r fast3r stream3r streamvggt monst3r vggt_omega; do
      python experiments/pilot_a.py --gt data/gt/hydrant.npz --backbone $bb \
          --chunk 16 --overlap 4 --json outputs/reports/pilotA_hydrant_$bb.json
    done

One process per backbone is required, not stylistic: six of these repos
vendor forks of dust3r/croco and shadow each other in a shared process.
That is 18 rows — the table Mengmi asked for.

Start with vggt and dust3r. If the mechanism does nothing on those two,
running the other seven changes nothing except the clock.

--------------------------------------------------------------------
## Step 4 — the length sweep, which is where the claim lives or dies
--------------------------------------------------------------------
    for n in 40 80 160 320; do
      python experiments/pilot_a.py --gt data/gt/hydrant.npz --backbone vggt \
          --chunk 16 --overlap 4 --max-frames $n \
          --json outputs/reports/pilotA_len$n.json
    done

SMR pays a fixed cost (the ~0.003 decode floor) and buys bounded drift
accumulation. So the honest prediction is: WORSE on short sequences,
crossing over somewhere, better beyond. If ATE-baseline / ATE-SMR rises
with length, the mechanism is real and this plot is the paper's spine. If
the ratio is flat, it is not, and we say so and pivot — that is what the
Sept 7 gate is for.

--------------------------------------------------------------------
## What I expect, stated in advance so it cannot be rationalised later
--------------------------------------------------------------------
  * short (<= 50 frames): SMR slightly WORSE. Little drift to correct, and
    the decode floor still costs.
  * long (200+): baseline ATE should grow faster, because chained Sim(3)
    compounds while the lattice snaps.
  * revisit column: the best chance of a clean win, since chunk-chaining
    has no mechanism to close a loop and memory does.
  * AUC@30 on short windows: near-parity at best. Report it anyway.

If the numbers disagree with all of that, the numbers are right.
