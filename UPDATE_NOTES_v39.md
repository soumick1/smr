# Update v3.9 -- they were never cloned (my setup script's fault)

## What happened
`setup_backbones.sh` defaulted to `group1`, which is dust3r/mast3r only.
fast3r and STream3R were never fetched, so "missing module" was literally
true and completely unhelpful.  Two fixes:

1. The error now distinguishes the two cases and names the command:
       fast3r: repository not cloned -- expected third_party/fast3r. Run:
           bash server/setup_backbones.sh fast3r
   versus, when the repo IS there but the package is not installed:
       ... exists but module(s) [...] are not importable ... Run:
           pip install -e third_party/fast3r --no-deps

2. setup_backbones.sh rewritten: named `group1` / `group2` / `all` targets,
   prints usage when called with no arguments, and installs with --no-deps.

## READ THIS BEFORE INSTALLING GROUP 2
fast3r's requirements.txt pins **numpy<2.0.0**.  Your environment has numpy
2.5.2 and VGGT, pi3, DUSt3R, MASt3R and our own code all run on it.
`pip install -r third_party/fast3r/requirements.txt` would downgrade numpy
and can break the four backbones that currently pass.

The script therefore installs the packages with `--no-deps` and adds only
what the inference path actually needs (lightning + einops -- both repos
are Lightning projects).  If something is still missing at [import], add it
one package at a time; do not run their requirements files wholesale.

## Commands
    bash server/setup_backbones.sh group2      # fast3r stream3r cut3r mvdust3r
    # weights: fast3r and stream3r pull their own from the HF hub on first
    # use; mvdust3r has a direct URL; cut3r needs gdown (printed by the
    # script).
    python scripts/fetch_weights.py --backbones mvdust3r

    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r --max-views 4

If lightning pulls a torch it does not like, install it with --no-deps too
and add only what it complains about -- protecting torch 2.5.1 and numpy 2
matters more than a tidy dependency tree.

Tests: 86 passed, 8 GPU-tier skipped.

## Reminder on priorities
Four backbones already bind, spanning the whole family tree.  That is
enough for Table 1.  31 days to the ICLR abstract, still zero tables --
after this run, Pilot A.
