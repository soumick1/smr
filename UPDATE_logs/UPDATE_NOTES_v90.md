# smr_updates90 — the VGGT-suite pivot, piece 1: Table-1 pose with memory consensus

    scp/unzip; python -m pytest -q     # expect 170 passed, 23 skipped
    git add -A && git commit -m "v90: context-consensus pose row for the VGGT Table-1 protocol; CO3D prep" && git push

## What this is
Mengmi's bar, honestly translated: memory must improve the backbone on its
OWN benchmarks.  Table 1 (CO3Dv2/RE10K pose AUC@30) is the flagship: VGGT
has a first-frame reference convention and is order-sensitive (pi3's whole
pitch).  The +SMR row runs k permuted passes over the SAME protocol frames,
binds them, and reads each pose out by robust cross-context consensus --
training-free, protocol-compliant, and a provable no-op for a
permutation-equivariant backbone (the pi3 row is the honesty check).

## Data (CO3D v2; start with 10 categories, ~15 GB)
    mkdir -p ~/co3d && cd ~/co3d
    # per-category zips from the official CO3D repo release page; each has
    # frame_annotations.jgz + images.  Start: apple bench book chair cup
    # hydrant laptop plant teddybear vase
    cd ~/smr && source .venv/bin/activate
    python scripts/co3d_eval_prep.py --root ~/co3d \
        --categories apple,bench,book,chair,cup,hydrant,laptop,plant,teddybear,vase \
        --n-frames 10 --max-seq 20 --out-dir data/gt/co3d

## Run (each backbone: 200 seqs x 8 passes x ~1.5 s ~= 40 min; passes cached)
    for bb in vggt vggt_omega pi3; do
      CUDA_VISIBLE_DEVICES=0 python experiments/vggt_suite.py --gt-glob 'data/gt/co3d/*.npz' \
        --backbone $bb --k-passes 8 \
        --json outputs/reports/suite_pose_co3d_${bb}.json 2>&1 | tee -a outputs/reports/suite_pose.log
    done

## Predictions, recorded
* vggt / vggt_omega: consensus AUC@30 exceeds single by +0.5 to +1.5 points
  (converted order-variance); AUC@15 gains larger than AUC@30.
* pi3: consensus == single within +-0.2 (permutation-equivariant; this row
  failing to move is evidence the gain is real, not a fusion artifact).
* If the vggt gain is < +0.3, the honest conclusion is that Omega's
  training already removed the order sensitivity, and Table 1 joins
  ScanNet-1500 as a memory-inert protocol -- reported as such.
