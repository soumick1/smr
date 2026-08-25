# Update v5.5 -- torch.load, generalised (and the RoPE2D warning)

## The checkpoint error: my shim was too narrow
CUT3R's checkpoint pickles `omegaconf.dictconfig.DictConfig`; DUSt3R's
pickles `argparse.Namespace`; the next one will pickle something else.
Allowlisting them one at a time cannot win, because we do NOT control the
call sites -- `from_pretrained` and friends call torch.load internally, so
there is no place to pass weights_only=False.

`patch_torch_load_legacy_checkpoints()` now does both:
  1. allowlists the config classes we know about (argparse.Namespace plus
     omegaconf DictConfig/ListConfig/ContainerMetadata/Metadata/AnyNode),
     for anyone who passes weights_only=True explicitly; and
  2. restores the pre-2.6 DEFAULT for calls that do not specify it.

The trust basis for (2) is narrow and written into the docstring: every
checkpoint we load came from the official URL recorded on its adapter
(weights_url / weights_hub / weights_gdrive) -- exactly the "you got the
file from a trusted source" condition PyTorch's own error describes.  An
explicit weights_only=True still wins.  Idempotent; no-op on torch < 2.6.
cut3r now applies it before loading.

## "cannot find cuda-compiled version of RoPE2D"
Benign.  croco ships an optional CUDA kernel for its 2-D rotary embeddings;
without it you get a correct pure-PyTorch fallback, just slower.  It
affects the RoPE-based models (dust3r, mast3r, monst3r, cut3r, pi3).

To build it, once per repo that vendors croco:

    cd third_party/dust3r/croco/models/curope
    python setup.py build_ext --inplace
    # repeat for: mast3r/dust3r, monst3r, CUT3R/src, streamvggt/src

Needs nvcc matching your torch CUDA build.  MY ADVICE: skip it for now.
It is a modest end-to-end speedup (RoPE is a small share of ViT time), it
must be repeated per vendored copy, and a failed build mid-deadline is
exactly the kind of distraction that has cost us today.  Revisit it if
Pilot A turns out to be compute-bound.

Tests: 114 passed, 11 skipped.
