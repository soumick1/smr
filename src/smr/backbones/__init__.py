from .base import Backbone, BackboneOutput, get_backbone, register  # noqa: F401
from .pointmap import PointmapBackbone, RawViews, assemble          # noqa: F401
from . import synthetic, vggt, pi3                                  # noqa: F401
from . import dust3r, mast3r, monst3r                               # noqa: F401
from . import fast3r, stream3r, streamvggt, vggt_omega              # noqa: F401
# Removed, with reasons (all upstream/environment, none of them our socket):
#   cut3r    -- its pose token uses position -1, which only works with
#               croco's compiled CUDA kernel; that build needs a CUDA
#               toolkit (CUDA_HOME/nvcc) absent from this machine.  The
#               adapter is complete and in git history.
#   must3r   -- needed a torchvision newer than the venv had at the time
#               (moot after the environment fix; ~15 min to restore).
#   mvdust3r -- its losses.py imports pytorch3d (build-from-source).
# Not a backbone:
#   lagernvs -- a novel-view-synthesis renderer that USES VGGT for poses;
#               it consumes geometry rather than producing it, so it is a
#               Table B baseline, not a socket member.
