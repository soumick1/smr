# Update v3.3 -- the three import failures, diagnosed and fixed

## Why dust3r / mast3r / must3r reported "missing module"

They were cloned fine.  The problem is that **DUSt3R and MASt3R ship no
setup.py and no pyproject.toml** -- only requirements.txt -- so the
`pip install -e` in setup_backbones.sh could never succeed.  Their own
demos just run from the repo root, i.e. they are PYTHONPATH-style repos.
(MASt3R also vendors dust3r as a submodule at mast3r/dust3r.)  MUSt3R does
have a setup.py, but there is no reason to treat it differently.

FIX: the adapters now bootstrap themselves.  Each declares which
third_party subdirectories it needs and inserts them on sys.path before
importing, idempotently:

    dust3r : ("dust3r",)
    mast3r : ("mast3r", "mast3r/dust3r", "dust3r")   # vendored copy first
    must3r : ("must3r", "must3r/dust3r")

preflight() bootstraps before testing importability, so the report no
longer shows a false negative.  Nothing to install, nothing to export --
re-run the checker and they should get past [import].

## pi3 FAIL at [infer], also fixed

pi3's loader takes a DIRECTORY and globs it in sorted order.  Passing 4
paths from a 41-image folder made it load all 41 -- an assertion caught it
(good), but the real fix is that a list of paths must mean exactly those
paths, in that order, for EVERY backbone.  pi3 now stages the requested
subset into a temp directory of ordered symlinks.  This matters well
beyond the checker: every evaluation harness passes subsets.

## conv=? in the report

vggt and pi3 predate the socket and never declared pose_convention.  Now
they do -- vggt "w2c" (inverted), pi3 "c2w" (not inverted) -- so the report
prints the convention for all five, and a test enforces the declaration.
This is the single most dangerous piece of per-backbone knowledge; it now
lives on the class, not in a comment.

## Re-run
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r must3r --max-views 4

Expected: vggt still PASS; pi3 PASS; dust3r/mast3r/must3r now reach
[load] or beyond.  DUSt3R will be slow (pairwise: O(K^2) inferences plus a
300-iteration global alignment) -- 4 views is the right size for a check.
If MUSt3R fails at [infer], paste the error: its inference return container
is the one API I could not fully pin from the repo, and the adapter is
written to fail loudly with a description of what it actually found.

Tests: 53 passed, 6 GPU-tier skipped.
