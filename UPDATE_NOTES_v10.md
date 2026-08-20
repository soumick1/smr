# Update v1.0 -- pi^3 adapter (written from the cloned official repo)

## Fixed first: the artifact-overwrite bug you caught
Reports/figures were tagged with the frames folder's LEAF name -- every
LLFF scene ends in images_8, so all eight runs collided.  Tags now use
the scene directory (verified: sceneA/sceneB -> distinct reports).
PLEASE RE-RUN the 8-scene sweep after applying; all JSONs will survive
and I will build the paper table from them:
    for s in fern flower fortress horns leaves orchids room trex; do \
      python experiments/run_tier3_vggt.py --backbone vggt \
             --frames lab_photos/$s/images_8; done

## pi^3 adapter (src/smr/backbones/pi3.py)
Written against the actual cloned source, not memory:
* camera_poses are CAMERA-TO-WORLD (OpenCV) = our convention: NO
  inversion (opposite of VGGT -- the classic silent bug, dodged).
* depth = local_points[..., 2]; conf is raw logits -> sigmoid, thr 0.1
  (their example), optional depth_normal_edge filter as in example.py.
* Intrinsics are NOT output; estimated per view by least squares from
  local points -- estimator unit-verified offline (exact on clean
  pinhole, ~0.2 px focal error at 1% depth noise).
* Preprocessing goes through THEIR load_images_as_tensor (sorted dir,
  uniform resize to <=255k px in multiples of 14) so it is faithful by
  construction; adapter asserts the directory matches the passed paths.
* variant="pi3x" (their Recommended) default; "pi3" and --ckpt paths
  supported exactly as in example.py.
server/setup_backbones.sh now clones the verified URL + installs.

## Then: the drift-comparison runs (zero harness changes)
    python scripts/vggt_sanity.py --frames outputs/synth_frames --backbone pi3
    python scripts/llff_pose_check.py --scene lab_photos/room --backbone pi3
    python experiments/run_tier3_vggt.py --backbone pi3 \
           --frames lab_photos/room/images_8
Expectation stated up front: pi^3's design goal is permutation
equivariance -- its reference-anchored-drift curve may be FLAT where
VGGT's rises.  Either outcome is the interesting figure: flat validates
their claim and positions SMR's consistency machinery as the fix for
VGGT-class backbones; rising would be a finding against theirs.
