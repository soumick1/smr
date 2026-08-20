"""vggt_omega adapter (server-side): lazy torch import; weights and repo are
installed by server/setup_backbones.sh.  The adapter's only job is to fill
BackboneOutput; everything downstream is untouched (plan, Sec. Portability).
"""
from __future__ import annotations

from .base import Backbone, BackboneOutput, register


@register("vggt_omega")
class Adapter(Backbone):
    name = "vggt_omega"

    def __init__(self, device="cuda", weights=None):
        self.device, self.weights = device, weights
        self._model = None

    def _load(self):
        try:
            import torch  # noqa: F401
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "torch not installed. Run setup_env.sh --with-backbones on "
                "the GPU server, then server/setup_backbones.sh.") from e
        raise NotImplementedError(
            "Wire the official vggt_omega repo here after cloning it with "
            "server/setup_backbones.sh (fill forward pass + output mapping).")

    def infer(self, images):                            # pragma: no cover
        if self._model is None:
            self._load()
