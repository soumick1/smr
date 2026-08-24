# Update v2.6 -- completion wired into the flagship

run_orbit_demo.py gains --complete <ckpt>: runs the trained head on the
arrival splat, hard-composites (known pixels untouched), adds the
"completed" panel to the figure, prints + reports completed rel-depth
at ~full coverage and hole-PSNR vs the actual held-out photo.
(Torch path first executes on your GPU, as usual -- the numpy/figure
path is smoke-tested here.)

Also included: train_completion.py with the eval-sample spread patch
(samples now drawn from across the val set, not four near-clones of
the alphabetically first scene).

## THE run (the before/after disocclusion figure, on your condo orbit)
    CUDA_VISIBLE_DEVICES=0 python experiments/run_orbit_demo.py \
        --backbone vggt --frames orbit_frames \
        --complete outputs/logs/completion/co3d_v1/ckpt_best.pt
    # then the orbit-path teaser with completion:
    CUDA_VISIBLE_DEVICES=0 python experiments/run_orbit_demo.py \
        --backbone vggt --frames orbit_frames --path orbit \
        --complete outputs/logs/completion/co3d_v1/ckpt_best.pt
Send the figures + JSONs; the completed panel goes straight into the
paper as fig:completion-flagship.
