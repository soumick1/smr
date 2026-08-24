# Update v3.7 -- the real finding: your torchvision is far older than torch

## Diagnosis (this one matters beyond MUSt3R)
Look at the warning printed on EVERY run since the beginning:

    torchvision/transforms/functional.py:63 ...
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

That code was removed from torchvision years ago.  Combined with
`center_crop` living at line 261, your installed torchvision is roughly
0.6-0.9 era while your torch is 2.5.1.  In that vintage, center_crop is
PIL-only:

    image_width, image_height = img.size

On a torch.Tensor `.size` is a METHOD, hence
"cannot unpack non-iterable builtin_function_or_method object".
MUSt3R's loader does CenterCrop(ToTensor(image)) -- so it is the first
backbone to touch that path.  The other four never do, which is why they
pass.

## Two ways forward
1. PROPER FIX (recommended, but AFTER the pilot experiment):
       pip install --upgrade "torchvision"   # match torch 2.5.1 -> 0.20.x
   This also clears the deprecation warnings VGGT/pi3 print.  It is an
   environment change, so do not do it the week of a deadline without
   re-running scripts/check_backbones.py afterwards.

2. NO-RISK SHIM (in this update, active now):
   `patch_torchvision_center_crop()` delegates PIL inputs to the original
   and slices tensors directly.  It self-checks first -- if your
   torchvision is already tensor-capable it does nothing -- so it becomes
   a no-op the moment you upgrade.

Crop geometry is unit-tested against torchvision's exact rounding,
including the banker's-rounding case (a 9->4 crop starts at 2, not 3);
getting that wrong shifts every image by a pixel, silently.

## Honest recommendation on time
4/5 backbones bind, and the four that work span the whole family tree
(VGGT feed-forward multi-view, pi3, DUSt3R pairwise+global-alignment,
MASt3R sparse-GA).  MUSt3R is experimental, non-commercially licensed, and
adds little the others do not already represent.  If this run still fails,
my advice is to mark it experimental, move to Pilot A, and revisit MUSt3R
after the Sept 7 gate -- the paper is decided by the long-sequence
experiment, not by the fifth backbone.

Tests: 69 passed, 6 GPU-tier skipped.
