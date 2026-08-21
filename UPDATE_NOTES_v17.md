# Update v1.7 -- the completion head: model + trainer

## What ships (and what was verified where)
* src/smr/completion/model.py -- CompletionUNet: fully-convolutional,
  5ch in (splat rgb + depth/4 + mask) -> rgb(sigmoid) + depth(softplus),
  base 48 (~12M params), 4 downs (eval pads to multiple of 16).
* experiments/train_completion.py -- the trainer.  Torch-free layers
  (dataset, SCENE-level split, stats, logging, figures) are verified
  locally against real harvested pairs; the torch loop (AMP, cosine+
  warmup, grad-clip, checkpointing, full-frame eval) is code-reviewed
  and gets its first execution in your 3-minute sanity run below.

## Logging (screen-safe, everything under outputs/logs/completion/<name>/)
  train.log    timestamped, mirrors console -- tail -f this
  metrics.csv  per-log-step: loss components, grad-norm, img/s
  eval.csv     per-eval: psnr_all, psnr_hole, l1_hole, absrel_hole/known
  config.json  args + git commit + dataset stats (coverage, context hist)
  samples/     qualitative grids each eval: splat|pred|composite|target|err
  ckpt_last.pt / ckpt_best.pt (best = psnr_hole); --resume continues

## Run ladder (in order)
0) Enrich the shakedown set with the 1-view/4-view regimes (resume-safe;
   complements the existing all-context pairs without duplicating them):
     for s in fern flower fortress horns leaves orchids room trex; do \
       python scripts/harvest_completion_pairs.py \
              --frames lab_photos/$s/images_8 --backbone vggt \
              --out data/completion --context 1,4; done
     python scripts/harvest_completion_pairs.py --frames orbit_frames \
            --backbone vggt --out data/completion --context 1,4
   (~2x the original harvest time; screen.)
1) Instant pipeline check on real data:
     python experiments/train_completion.py --data data/completion \
            --name shakedown --val-scenes room,orbit_frames --dry-run
2) 3-minute GPU sanity (first execution of the torch loop):
     python experiments/train_completion.py --data data/completion \
            --name shakedown_sanity --val-scenes room,orbit_frames \
            --steps 300 --eval-every 100
   Expect: loss falling from ~0.15-0.3, an EVAL line at 100/200/300,
   samples/*.png appearing, ~40-80 img/s on the A6000.
3) The shakedown train (screen):
     python experiments/train_completion.py --data data/completion \
            --name shakedown --val-scenes room,orbit_frames --steps 8000
   ~20-40 min.  After a disconnect: same command + --resume.
4) Paste train.log's tail + eval.csv + a couple of samples/ images.

Next session: --complete <ckpt> wired into run_orbit_demo (the
before/after disocclusion figure), then CO3D at scale via harvest_root.
