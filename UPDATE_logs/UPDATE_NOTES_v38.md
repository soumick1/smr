# Update v3.8 -- MUSt3R removed, group 2 added (7 backbones registered)

## MUSt3R: removed, and why it was never going to work here
The last error (`resize: img should be PIL Image. Got torch.Tensor`, at
functional.py:182) confirms the diagnosis: in your torchvision vintage
EVERY transform is PIL-only, so shimming them one at a time is a losing
game.  Removing it is right.  Note it is a ~15-minute restore later:
upgrade torchvision to match torch 2.5.1 and re-add the adapter from git
history -- nothing about our socket was wrong.

ON THE SERVER, to reclaim the space:
    rm -f third_party/checkpoints/MUSt3R_512*.pth
    rm -rf third_party/must3r
    git rm src/smr/backbones/must3r.py     # already unregistered by this zip

## Group 2 adapters (APIs read from each repo, not memory)
* fast3r    c2w  hub jedyang97/Fast3R_ViT_Large_512
            load_images -> inference -> MultiViewDUSt3RLitModule
            .estimate_camera_poses(...).  Their point maps are GLOBAL
            ('pts3d_in_other_view'), so the adapter transforms them into
            each camera's frame before the socket sees them.
* stream3r  w2c  hub yslan/STream3R
            VGGT-derived: model(images, mode="causal"|"window"|"full");
            keys pose_enc / depth / depth_conf / world_points.  Default
            mode is "causal" -- the streaming setting their paper is about,
            and exactly the regime our long-sequence experiment targets.
* cut3r     c2w  EXPERIMENTAL, weights are Google-Drive only:
                pip install gdown && cd third_party/checkpoints && \
                gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view
            demo.py sits at their repo root, so we sys.path it and import
            prepare_input / prepare_output directly.
* mvdust3r  c2w  EXPERIMENTAL, weights fetchable (HF resolve URL for
            MVDp_s1.pth).  Its inference entry point is a demo script
            rather than a library call; `_raw` raises a precise message
            asking for that one signature.  Everything else is wired.

## Registered now
    vggt(w2c) pi3(c2w) dust3r(c2w) mast3r(c2w)
    fast3r(c2w) stream3r(w2c) cut3r(c2w) mvdust3r(c2w)

## Run
    python scripts/fetch_weights.py --list        # now shows hub/manual too
    python scripts/fetch_weights.py --backbones mvdust3r
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r --max-views 4
(fast3r and stream3r download their own weights from the hub on first use.)
Add cut3r once gdown has fetched its checkpoint.

Tests: 80 passed, 8 GPU-tier skipped.

## Next, and I mean this
Six backbones behind one socket is already enough for Table 1.  The clock
says 31 days to the ICLR abstract and there are still zero tables.  The
next session should build Pilot A (chunked long sequences, SMR stitching vs
Sim(3) chaining, ATE/RPE + peak memory), not backbone number nine.
