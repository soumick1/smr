# Update v5.8 -- cut3r removed. The socket is DONE: nine backbones.

cut3r is unregistered (adapter deleted; it is complete and in git history
if a CUDA toolkit ever appears on that machine).  The reason is recorded in
backbones/__init__.py so nobody re-litigates it: their pose token uses
position -1, which only works with croco's compiled CUDA kernel, and that
build needs CUDA_HOME/nvcc which this server does not have.

ON THE SERVER:
    git rm src/smr/backbones/cut3r.py
    rm -f third_party/checkpoints/cut3r_512_dpt_4_64.pth   # ~2 GB back
    rm -rf third_party/CUT3R                                # optional

## The final socket
    vggt        w2c    feed-forward multi-view
    pi3         c2w    feed-forward, camera-to-world native
    dust3r      c2w    pairwise + global alignment
    mast3r      c2w    sparse global alignment
    fast3r      c2w    fast many-view
    stream3r    w2c    causal / streaming
    streamvggt  w2c    causal / streaming (independent design)
    monst3r     c2w    dynamic scenes
    vggt_omega  w2c    newest Omega model

Nine models, one interface, conventions declared on the class and enforced
by tests, no backbone-specific code anywhere downstream.  106 tests pass
without a GPU.

For the paper: state plainly which models were integrated and why the
others were not -- cut3r (compiled CUDA extension unavailable), mvdust3r
(pytorch3d), lagernvs (an NVS renderer that consumes geometry rather than
producing it, hence a Table B baseline).  Reviewers respect a stated
inclusion criterion far more than an unexplained sample.

## NEXT SESSION: PILOT A. Nothing else gets opened.
    * chunk CO3D sequences (already on the server) into windows of 16 with
      4-frame overlap
    * stitch two ways: Umeyama/Sim(3) chaining on the overlaps vs binding
      each chunk into the scaffold and reading the trajectory out
    * hierarchical: scaffold gives the coarse drift-free global frame, the
      backbone's within-chunk poses give local precision
    * report ATE-RMSE, RPE-trans, RPE-rot, peak GPU memory, time/frame
    * PASS: +SMR beats chained Sim(3) on ATE at >=100 frames on >=60% of
      sequences
    * run each backbone in its OWN PROCESS (six of these repos vendor forks
      of dust3r/croco -- today proved why)

27 days to the ICLR abstract.
