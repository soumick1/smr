"""pi3 adapter (server-side): lazy torch import; weights and repo are
installed by server/setup_backbones.sh.  The adapter's only job is to fill
BackboneOutput; everything downstream is untouched (plan, Sec. Portability).
"""
from __future__ import annotations

from .base import Backbone, BackboneOutput, register


@register("pi3")
class Adapter(Backbone):
    name = "pi3"

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
            "Wire the official pi3 repo here after cloning it with "
            "server/setup_backbones.sh (fill forward pass + output mapping).")

    def infer(self, images):                            # pragma: no cover
        if self._model is None:
            self._load()
