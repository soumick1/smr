# Update v5.1 -- tokenizers pin, seaborn, cut3r mirror

## 1. tokenizers -- ONE command, fixes fast3r AND streamvggt
`--no-deps` cut both ways: pip installed tokenizers 0.23.1 without checking
transformers' own pin (`tokenizers>=0.22.0,<=0.23.0`), so transformers now
refuses to import -- which broke fast3r too, since it imports transformers.

    pip install --no-deps "tokenizers>=0.22.0,<=0.23.0"

(setup_backbones.sh now states the pin explicitly.)

## 2. monst3r: seaborn -- fixed in code
`dust3r/cloud_opt/init_im_poses.py` imports seaborn at module level.  That
file IS on the inference path, but seaborn is used only for figures, so it
joins the allowlist: evo, seaborn, open3d, wandb.  Nothing to install.
Note the previous run shows the stub machinery worked exactly as intended
on evo -- it reported all seven evo submodules it faked.

## 3. cut3r: now points at your mirror
Their official Drive link is quota-blocked, so the adapter uses your copy
(1V7TEIpHncGBpgsSx8R2fxYISCgE2w9Bl), with the original recorded in a
comment.  Worth noting in the repo README that this is a mirror of the
official checkpoint, so reviewers can verify provenance.

    pip install --no-deps "tokenizers>=0.22.0,<=0.23.0"
    python scripts/fetch_weights.py --backbones cut3r
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r \
                    streamvggt monst3r vggt_omega cut3r --max-views 4

Tests: 111 passed, 10 GPU-tier skipped.
