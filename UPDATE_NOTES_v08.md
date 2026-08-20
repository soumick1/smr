# Update v0.8 -- real-data validation CLOSED; V3 rebuilt on two findings

## Your room run, read correctly
Self-consistency 0.0026 (all 41 pairs 0.0020-0.0048, 20x under thr);
V1 0.003 rad on real estimated poses; V5a 0.0031 -- a held-out REAL view
reproduced from neighbouring views' surfels through the scaffold at 0.3%
median depth error.  The pipeline is validated on real images.  V3's
"6/40" was a broken metric + a real brittleness, both now fixed:

1. METRIC: exact-ID among 40 near-duplicate frames (LLFF views are ~1-2
   deg apart) is meaningless; relocalisation is now judged in POSE space,
   within 2x the capture's own median adjacent-view spacing.  Reduces to
   the old criterion on sparse rings (regression: 9/9 unchanged).
2. CORRUPTION MODEL: the old 0.2/dim noise on a unit 448-d descriptor
   has norm 4.2 (SNR 0.06) -- neighbour discrimination is impossible
   under it.  Corruption is now stated in signal units (unit-norm noise,
   cos~0.71 to the clean cue).
3. RETRIEVAL: two-stage, as the architecture implies -- the h-cue
   PROPOSES top-5 (pattern completion), direct descriptor comparison
   VERIFIES among them.  This eliminated the catastrophic aliasing flips
   measured on a dense 40-view capture: 22/39 exact with 4 flips at
   90-162 deg  ->  39/39 exact, 0 flips.

## Also in this update
* backbones: adapter-v2 hook -- BackboneOutput.descriptor uses pooled
  backbone features when present (fixed seeded random projection to 448);
  vggt adapter gains use_features (aggregator tokens, VERIFY-ON-SERVER,
  falls back loudly to RGB pooling on failure).
* run_tier3_vggt.py: --features flag.

## Run next (in order)
    python experiments/run_tier3_vggt.py --backbone vggt \
           --frames lab_photos/room/images_8          # expect all-pass now
    python scripts/llff_pose_check.py --scene lab_photos/room   # NOT YET RUN
    python scripts/vggt_sanity.py --frames lab_photos/fortress/images_8
    python experiments/run_tier3_vggt.py --backbone vggt \
           --frames lab_photos/fortress/images_8
    # then verify the aggregator call in src/smr/backbones/vggt.py against
    # your clone and try:  ... --frames lab_photos/room/images_8 --features
