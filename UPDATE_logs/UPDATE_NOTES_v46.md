# Update v4.6 -- fast3r return shape

    ValueError: too many values to unpack (expected 2)

Read from their source (inference_multiview.py, lines 80-99): inference()
returns `(result, profiling_info)` ONLY when profiling=True AND profiling
info was produced; otherwise it returns the collated result dict directly.
The README's `output_dict, profiling_info = inference(...)` example passes
profiling=True, which is what my adapter copied while passing
profiling=False.

Fixed by taking the value and unwrapping only if it is a tuple.  Pinned by
a test so the README's shape cannot creep back in.

    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r stream3r fast3r --max-views 4

Tests: 94 passed, 8 GPU-tier skipped.

## If fast3r fails again, ship without it
Five backbones already pass and they span feed-forward multi-view, pairwise
+ global alignment, sparse GA, and causal streaming.  That is a complete
plug-in claim.  fast3r is a nice sixth row, not a requirement, and the
remaining risk is all in its Lightning wrapper rather than in our socket.

28 days.  Pilot A next, either way.
