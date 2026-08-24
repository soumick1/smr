# Update v1.6 -- training-corpus answer, in code

Your question was the right one, at the right time.  Decisions now
wired in:

1. TRAINING DATA (the A* answer): the 337 LLFF+orbit pairs are the
   pipeline-shakedown set ONLY -- the head trains on established
   corpora with scene-disjoint splits:
     * CO3D (object orbits, ~19k sequences) -- your capture regime at
       scale; downloadable now from the official repo.
     * ScanNet++ (indoor scenes; needs the access application -- start
       it this week) or RealEstate10K as the fallback.
   LLFF + 7-Scenes become ZERO-SHOT test sets (never trained on),
   which makes them stronger evidence, and your condo orbit is the
   in-the-wild demo.

2. ONE HEAD FOR 1-VIEW..N-VIEW (the OOD answer): everything except the
   head is training-free (scaffold/bind/drive operate on frozen
   backbone output, so generality is inherited); the head is the only
   component with a domain.  It now trains across the whole context
   spectrum: harvest_completion_pairs.py --context 1,4,all binds the
   nearest-1 / nearest-4 / all views per holdout (measured on the
   smoke: coverage 0.45 -> 0.57 with context, i.e. the 1-view regime
   has the big holes, as it must).

3. NEW scripts/harvest_root.py: dataset driver -- walks any root of
   scene dirs (CO3D/ScanNet++/RE10K layout), subsamples to
   --max-views, calls the harvester per scene, resume-safe.

## When CO3D lands on the server
    python scripts/harvest_root.py --root <co3d_frames_root> \
           --backbone vggt --out data/completion_co3d \
           --holdouts every:5 --context 1,4,all --max-views 32 \
           --limit-scenes 200        # start with 200 scenes (~1 day CPU)
Run inside screen.  Multi-scene retention (I7, remap offsets on) is the
next architecture milestone after the head -- designed and
source-measured, our own test still pending, tracked honestly.
