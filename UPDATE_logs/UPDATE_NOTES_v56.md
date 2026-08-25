# Update v5.6 -- cut3r: the real prepare_input / prepare_output signatures

The model loads; this was the last call-signature guess.  Read from
demo.py:

    prepare_input(img_paths, img_mask, size, raymaps=None,
                  raymap_mask=None, revisit=1, update=True)
    # their own main passes img_mask=[True] * len(img_paths)

    prepare_output(outputs, outdir, revisit=1, use_pose=True)
      -> (pts3ds_other, colors, conf, cam_dict)
      cam_dict = {"focal": (B,), "pp": (B,2), "R": (B,3,3), "t": (B,3)}

So three things changed in the adapter:
  * prepare_input is called with img_mask (the missing argument);
  * poses come straight from cam_dict R/t, which are ALREADY
    camera-to-world -- no guessing, and the speculative multi-key
    extractor is gone;
  * the focal is taken from cam_dict rather than estimated by the socket,
    since they compute it with estimate_focal_knowing_depth(weiszfeld);
  * use_pose=True returns world-frame points, so the adapter undoes each
    pose to hand the socket camera-frame point maps (same as fast3r).

Pose assembly is unit-tested, including the failure message when cam_dict
has an unexpected layout.

    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r \
                    streamvggt monst3r vggt_omega cut3r --max-views 4

Tests: 115 passed, 11 skipped.
