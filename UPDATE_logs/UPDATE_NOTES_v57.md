# Update v5.7 -- cut3r: diagnosed to the line; the fix is building curope

## Root cause (not our adapter, and not fixable from our side)

CUT3R gives its POSE token a negative sentinel position:

    src/dust3r/model.py:769   pose_pos_i = -torch.ones(...)
    src/dust3r/model.py:834   pose_pos_i = -torch.ones(...)

The pure-PyTorch RoPE2D fallback in croco looks positions up in a table:

    src/croco/models/pos_embed.py:154
        cos = torch.nn.functional.embedding(pos1d, cos)

`F.embedding(-1, table)` is an out-of-bounds gather -> the device-side
assert you saw.  (The reported frame, `int(positions.max())` on line 172,
is just where the async CUDA error surfaced, because .max() forces a sync.)

Their CUDA kernel does NOT do a table lookup -- it computes the rotation
analytically:

    src/croco/models/curope/kernels.cu:53
        const float freq = pos[...] * shared_inv_freq[...];
        const float cos = cosf(freq);   const float sin = sinf(freq);

cosf(-x) and sinf(-x) are perfectly well defined, so with the compiled
kernel the -1 sentinel works exactly as intended.  CUT3R was only ever
tested with curope built -- which is why the repo warns twice on every run.

## The fix: build curope for CUT3R (2 minutes)

    cd third_party/CUT3R/src/croco/models/curope
    python setup.py build_ext --inplace
    cd ~/smr
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones cut3r --max-views 4

The "cannot find cuda-compiled version of RoPE2D" warning for cut3r should
disappear, and with it the assert.  Needs nvcc matching your torch CUDA
build; if the compile fails, see below.

## What I will NOT do
Patch RoPE2D to clamp or wrap negative positions.  That would silently
change the model's semantics -- the pose token would land at position 0,
aliasing a real image patch -- and produce plausible-looking but wrong
numbers in a table we intend to publish.  A backbone that is honestly
absent beats one that is quietly wrong.

## If the build fails
Ship with 9/10.  VGGT, pi3, DUSt3R, MASt3R, Fast3R, STream3R, StreamVGGT,
MonST3R and VGGT-Omega all bind, spanning feed-forward multi-view,
pairwise+global-alignment, sparse-GA, two causal/streaming designs,
dynamic scenes and the newest Omega model.  CUT3R is one more row in a
table that is already the strongest plug-in claim in the literature, and
its absence costs a footnote, not the paper.
