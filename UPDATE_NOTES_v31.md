# Update v3.1 -- backbone socket + group 1 (DUSt3R / MASt3R / MUSt3R)

## What this is
The plug-and-play claim needs N backbones behind ONE socket.  This update
adds that socket and the first three new backbones, with tests that run
WITHOUT a GPU so the conversion can never silently drift.

## New: src/smr/backbones/pointmap.py -- the socket
Every model in the target list (DUSt3R, MASt3R, MUSt3R, CUT3R, Fast3R,
STream3R, MV-DUSt3R+, VGGT-Omega) emits the same triple: a pose, a local
point map (or depth), a confidence map.  `PointmapBackbone` implements the
conversion to BackboneOutput ONCE -- pose convention, focal estimation,
confidence masking, per-scene scale gauge -- so a new adapter supplies only
`_raw()` and DECLARES `pose_convention`.  Adding backbone number 9 is now a
~60-line job, not a new source of convention bugs.

Focal estimation is closed-form least squares in OUR projection convention
(u = fX/Z + W/2), so estimated intrinsics feed splat() directly.

## New adapters (APIs read from the actual repos, 2026-08-24, not memory)
* dust3r  -- load_images/make_pairs/inference/global_aligner; poses are
             CAMERA-TO-WORLD (verified: demo.py binds get_im_poses() to
             `cams2world`; base_opt does inv(get_im_poses()) for w2c).
* mast3r  -- sparse_global_alignment (its own recipe) with a dust3r
             global-aligner fallback via --aligner dust3r; c2w (`cam2w`).
             get_dense_pts3d() -> (pts3d, depthmaps, confs), depth flat.
* must3r  -- load_model + must3r_inference; per-view dicts with 'c2w',
             'pts3d_local', 'conf'.  Marked EXPERIMENTAL: the return
             container is version-dependent, so the adapter raises a loud
             descriptive error naming what it found rather than guessing.

## New: tests/test_backbones.py -- 30 tests, all green here, no GPU needed
Pins the things that have actually bitten us:
  * c2w vs w2c inversion (both directions, plus a test that the two
    declarations really differ -- so the check isn't vacuous)
  * focal recovered exactly from a point map, including with 40% holes
  * scale gauge: median confident depth == 1
  * conf_keep really controls the kept fraction
  * K=1 single view (the bug that broke the Gradio demo)
  * output schema: shapes, dtypes, rigid poses, descriptor size
  * THE COMPATIBILITY GUARANTEE: anything satisfying RawViews binds into
    the scaffold and self-recalls -- no backbone-specific code downstream
GPU tier (marked `gpu`, skipped by default):
    SMR_GPU_TESTS=1 SMR_TEST_FRAMES=orbit_frames pytest tests/test_backbones.py -m gpu -v

## New: scripts/check_backbones.py -- the report you asked for
Walks import -> load -> infer -> schema -> bind for each backbone and
prints a table plus JSON; nothing is fatal, so one run tells you the state
of the whole matrix.

    bash server/setup_backbones.sh            # clones+installs group 1
    python scripts/check_backbones.py --frames lab_photos/room/images_8
    python scripts/check_backbones.py --frames orbit_frames --verbose-errors

Expect DUSt3R/MASt3R to need weights downloaded on first run (HF hub) and
MASt3R's sparse aligner to want a cache dir (a temp dir is made for you).

## Licence note for the paper
dust3r and mast3r are CC BY-NC-SA 4.0 (non-commercial).  State it.

## Next session
Group 2 adapters (CUT3R, Fast3R, STream3R, MV-DUSt3R+) on the same socket,
then group 3 (VGGT-Omega, LaGeR-NVS).  Send me check_backbones.py's output
for group 1 first -- real API surprises are cheapest to fix one group at a
time.
