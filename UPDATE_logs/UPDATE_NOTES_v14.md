# Update v1.4 -- orbital mental path + full-scale flag

## New flags (both smoke-tested locally)
* run_orbit_demo.py --path orbit : waypoints the drive through the
  capture poses around the ring to the target -- the model mentally
  WALKS the orbit, object held in view, including across the held-out
  sector.  (Smoke: 1295 steps, arrival 0.0107/0.0185, splat 0.0098.)
  --path direct (default) remains the chord: imagination off the data
  manifold, through positions the camera never visited.
* run_orbit_demo.py / run_tier3_vggt.py --scale full : ring 256,
  torus 64, N_h 8192, k 410, spacing 0.3.  Smoke result: V1
  (0.0017, 0.0338) -> (0.0006, 0.0030) -- position decode floor down
  11x to 0.3% of scene scale, 66 s for a 10-view scene (~3x compact).

## Server runs (the server-scale milestone, now turnkey)
    # the teaser GIF: mental orbit of your machine
    python experiments/run_orbit_demo.py --backbone vggt \
           --frames orbit_frames --path orbit
    # full-scale on real data: floor should drop ~10x scene-wide
    python experiments/run_tier3_vggt.py --backbone vggt \
           --frames lab_photos/room/images_8 --scale full
    python experiments/run_orbit_demo.py --backbone vggt \
           --frames orbit_frames --scale full
    # and the full synthetic ladder at scale, if not yet done:
    python experiments/run_tier1_dynamics.py
    python experiments/run_tier2_memory.py
    python experiments/run_tier3_integration.py
