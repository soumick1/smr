# Update v4.0 -- fast3r's missing dep + MV-DUSt3R+ completed

STream3R passed first try (conv=w2c, f=392.21 -- within 0.1% of VGGT's
391.87, exactly as expected from a VGGT-derived camera head).  5/6.

## fast3r: one missing package
    ModuleNotFoundError: No module named 'omegaconf'
    at third_party/fast3r/fast3r/models/fast3r.py line 13

fast3r and STream3R are Lightning+Hydra projects; omegaconf is imported at
module level.  It is pure python with no numpy pin, so it is safe:

    pip install omegaconf hydra-core rootutils

(setup_backbones.sh now installs these alongside lightning + einops.)
If the next error names another package, add it ONE AT A TIME -- do not run
their requirements.txt, which pins numpy<2.0.0 and would downgrade the
numpy your four working backbones use.

## MV-DUSt3R+ : adapter completed (you have the weights)
Their `get_reconstructed_scene` cannot be called as a library function --
it contains a literal `input('press enter to continue')` and writes a .glb.
So the adapter reproduces the parts that matter, read from their demo.py:

  * model class is `AsymmetricCroCo3DStereoMultiView` (NOT the plain
    AsymmetricCroCo3DStereo my scaffold assumed) via `inference_mv`
  * point maps: pred1['pts3d'] + [x['pts3d_in_other_view'] for x in pred2s],
    all in the FIRST view's frame -> transformed per-camera for the socket
  * focal: estimate_focal_knowing_depth on the first view's top-3% confident
    pixels (their exact recipe)
  * poses: calibrate_camera_pnpransac per view against that shared pinhole
    -> `cams2world`, i.e. c2w
  * their loader takes an n_frame argument; we pass it and fall back if the
    signature differs

## Run
    pip install omegaconf hydra-core rootutils
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r mvdust3r \
        --max-views 4

Tests: 86 passed, 8 GPU-tier skipped.

## After this run
Six or seven backbones behind one socket is more than Table 1 needs.
30 days to the ICLR abstract, zero tables.  Next session: Pilot A.
