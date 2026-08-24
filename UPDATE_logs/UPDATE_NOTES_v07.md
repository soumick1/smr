# Update v0.7 -- LLFF real-image evaluation

You downloaded the LLFF benchmark (better than lab photos: curated
single-scene handheld captures WITH COLMAP ground-truth poses).  Two
things to know:

1. The existing scripts need a FLAT folder of ONE scene's images:
       python scripts/vggt_sanity.py --frames lab_photos/room/images_8
       python experiments/run_tier3_vggt.py --backbone vggt \
              --frames lab_photos/room/images_8
   Never point them at the parent dir (eight scenes = the poppy-field
   failure again).  Recommended order: room -> fortress -> fern.
   Use images_8; expect fern's V5a hotter (thin foliage = real parallax).

2. NEW scripts/llff_pose_check.py: VGGT vs COLMAP GT on real images:
       python scripts/llff_pose_check.py --scene lab_photos/room
   Headline metrics are convention-robust by construction (Sim(3)-aligned
   centre errors; relative-rotation magnitudes -- invariant to camera-axis
   permutations); absolute rotation is reported separately under the
   documented LLFF->OpenCV conversion and labelled VERIFY.  The converter
   itself is round-trip verified offline (max err 1e-16) and prints GT
   self-diagnostics (forward-facing captures must show small view-dir
   spread) so a broken conversion is caught before VGGT is blamed.
