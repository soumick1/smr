# Update v0.9 -- results recorded; harness ready for pi^3

## Recorded in the plan (smr_overleaf.zip recompiled, 41 pp, 0 errors)
New Sec. "Measured status (implementation v0.2)" before Gates: the 25/25
ladder values, estimated-geometry results (incl. the 5.1%-under-28.9deg
local-consistency finding as the "before" curve for pi^3), the LLFF
room/fortress real-data table (SC 0.0026/0.0036, held-out 0.3%, reloc
40/40 and 41/41, VGGT-vs-COLMAP centre 0.26% / rel-rot 0.023 deg with
the 1.3-deg alignment-gauge caveat), and the revised retrieval protocol
(signal-unit corruption, pose-space criterion, propose-verify).

## Changed code
* scripts/vggt_sanity.py, scripts/llff_pose_check.py -- --backbone flag:
  once the pi^3 adapter is wired, the entire real-data harness (sanity,
  pose check, tier-3) runs on it unchanged, and the drift comparison is
  just the same commands with --backbone pi3.

## Your next runs
1. Adapter v2, one command after checking the aggregator call in
   src/smr/backbones/vggt.py against your clone:
       python experiments/run_tier3_vggt.py --backbone vggt \
              --frames lab_photos/room/images_8 --features
2. Full LLFF sweep for the paper table (one line):
       for s in fern flower fortress horns leaves orchids room trex; do \
         python experiments/run_tier3_vggt.py --backbone vggt \
                --frames lab_photos/$s/images_8; done
   (horns/trex are heaviest; subsample every 2nd frame if VRAM protests.)
3. pi^3: clone the verified official repo into third_party/, fill
   src/smr/backbones/pi3.py to the BackboneOutput contract (camera-from-
   world -> invert; per-scene median-depth := 1; conf mask), then rerun
   the same three commands with --backbone pi3 -- the reference-drift
   panel gains its second curve with zero harness changes.
