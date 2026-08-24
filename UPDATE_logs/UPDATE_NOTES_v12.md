# Update v1.2 -- the flagship on your video: orbital mental rotation

Your capture is GOOD: full loop around the machine, object centred,
uniformly sharp (no blur-suspect frames at 4K), texture-rich target.
Thin bars will run the splat error hotter at silhouettes -- expected,
same as trex.

## New
* scripts/video_to_frames.py -- video -> frames dir: ffprobe-timed
  sampling at 3x the target count, sharpest-per-group selection,
  upright orientation, max-dim 1024.  (Tested on your actual video:
  32 clean frames.)
* experiments/run_orbit_demo.py -- THE demo, GT-free: bind all views
  except a held-out target ~120 deg around the orbit, place at the
  start view, physically drive the bumps to the target pose, then:
  arrival pose error vs the backbone's own pose for that view;
  V5-style splat-at-decoded-arrival vs the actual held-out frame;
  4-panel figure incl. top-down mental path over the capture ring;
  sweep GIF (recalled render + live yaw ring + torus) -- mental
  rotation of a REAL object, made visible.
  Smoke-tested end-to-end on a synthetic capture: 1017 steps, arrival
  0.0085 rad / 0.019, splat 0.0102 at 94% coverage.

## Run (after scp-ing the video to ~/smr)
    python scripts/video_to_frames.py --video 20260821_111236.mp4 \
           --out orbit_frames --n 32
    python scripts/vggt_sanity.py --frames orbit_frames
    python experiments/run_orbit_demo.py --backbone vggt --frames orbit_frames
    python experiments/run_orbit_demo.py --backbone pi3  --frames orbit_frames
    python experiments/run_tier3_vggt.py --backbone vggt --frames orbit_frames

## What to expect
A full orbit is the drift-stress test for a reference-anchored backbone:
VGGT's self-consistency may run above its LLFF numbers, and that is the
point -- the pi^3 run alongside it is the sharpest equivariance
comparison we can produce, now on real data and on the flagship task.
The arrival metric is measured against the backbone's OWN pose for the
held-out view, so global drift does not contaminate it (the
local-consistency argument, applied deliberately).  Paste both orbit
JSONs and the GIFs.
