# Update v5.2 -- the last two: monst3r/sam2 and cut3r's package path

8/10 -> both remaining failures are fixed in code; nothing to install.

## monst3r: No module named 'sam2'
`cloud_opt/optimizer.py` imports SAM2 at MODULE level (line 14) but only
CALLS it from `refine_motion_mask_w_sam2()` (line 367), reached when the
optimizer's `sam2_mask_refine` flag is true -- and their default is true.

So two changes, together:
  * sam2 joins the stub allowlist, and
  * the adapter passes `sam2_mask_refine=False` to global_aligner.
Stubbing a real model is only honest if nothing invokes it, so a test now
enforces that link: monst3r must default the flag off, and the flag must
reach global_aligner.  `get_backbone("monst3r", sam2_mask_refine=True)`
still opts in if you install SAM2 and want their dynamic-mask refinement
(their download_ckpt.sh fetches sam2.1_hiera_large.pt).

## cut3r: No module named 'dust3r.utils.camera'
The vendored-fork trap once more, with a twist.  Their
`add_ckpt_path.add_path_to_dust3r(ckpt)` inserts the CHECKPOINT'S OWN
DIRECTORY on sys.path -- which tells you their code refers to the fork as
top-level `dust3r`, not `src.dust3r`, and expects CUT3R/src on the path.

Fixed: third_party_paths is now ("CUT3R/src", "CUT3R") -- src leads so the
fork owns the `dust3r` name, the repo root follows for `import demo` -- and
both _load and _raw call isolate_third_party() first so a plain dust3r
cached by an earlier backbone in the same process cannot shadow it.

## Run
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r \
                    streamvggt monst3r vggt_omega cut3r --max-views 4

Tests: 112 passed, 10 GPU-tier skipped.

## Whatever this reports, the socket is finished
Eight already bind.  Ten adapters, conventions declared and tested, one
interface, no backbone-specific code downstream.  Next session is Pilot A
and I will not open another adapter until there is a table.
