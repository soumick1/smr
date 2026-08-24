# Update v3.5 -- MUSt3R: the asmk import, properly bypassed

4/5 -> 5/5 (expected).  MASt3R now passes at coverage 1.0.

## What the asmk error actually was
My faiss stub worked -- the except-branch never ran.  The real blocker is
must3r/retrieval/processor.py line 29:

    from asmk import asmk_method        # module level, unconditional

so the whole retrieval module needs asmk regardless of faiss, and asmk is
not a pip package: their README has you git-clone jenicek/asmk, cythonize
it and build.  Meanwhile must3r/demo/inference.py imports Retriever at
module level, so importing the inference entry point drags all of it in --
for a code path (unordered retrieval) that sequence mode never touches.

## The fix: stub the module, not its dependency tree
`_import_inference()` now tries the REAL import first, so a complete
installation is used untouched.  Only on ImportError does it stub
`must3r.retrieval.processor` -- a single module exposing an unused
`Retriever` -- and retry.  That is one honest stub instead of faking both
faiss and asmk, and it prints exactly what it did and why.

`ensure_module_stub` gained dotted-name support: it binds the stub onto the
parent package as well, so `from must3r.retrieval.processor import Retriever`
resolves.

If you ever want MUSt3R's unordered mode, follow their README (faiss-cpu +
building asmk) and the stub silently stops being used -- there is a test
asserting the real import is attempted first.

## Re-run
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r must3r --max-views 4

If MUSt3R now fails, it should finally be at the one thing I could not pin
from the repo -- the shape of must3r_inference's return value -- and the
adapter prints what it actually found.

Tests: 65 passed, 6 GPU-tier skipped.
