"""VGGT-Omega retrained checkpoint (v178/v180, plan Task 1).

facebookresearch/vggt-omega/reproduction.md: after a contamination review the retrained checkpoint (Aug 2026)
supersedes the original as the reference for benchmark comparisons and must be preprocessed with
image_resolution=416.  The file on the Hub is listed as `vggt_omega_1b_512_reproduce_noconf.pt` (4.58 GB; the
reproduction notes call it `vggt_omega_1b_512_reproduce.pt`), so any local `vggt_omega_1b_512_reproduce*.pt` is
accepted.  Registered under its OWN backbone name so every pass cache, report row and provenance record is distinct
from the original checkpoint: the two are never mixed.

    hf download facebook/VGGT-Omega --include "vggt_omega_1b_512_reproduce*" --local-dir third_party/checkpoints
    python experiments/pilot_a.py --backbone vggt_omega_repro ...
"noconf" may mean the confidence head was not trained/exported: the state dict is loaded strictly first and
non-strictly (with a printed list of missing keys) only if strict loading fails, and a missing `depth_conf`
output is replaced by ones (every point kept) with a one-time notice.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from .base import register
from .pointmap import RawViews, checkpoint_dirs, isolate_third_party
from .vggt_omega import VGGTOmegaBackbone


@register("vggt_omega_repro")
class VGGTOmegaReproBackbone(VGGTOmegaBackbone):
    name = "vggt_omega_repro"
    weights = "vggt_omega_1b_512_reproduce_noconf.pt"
    weights_file = "vggt_omega_1b_512_reproduce_noconf.pt"
    weights_url = ("https://huggingface.co/facebook/VGGT-Omega/resolve/main/"
                   "vggt_omega_1b_512_reproduce_noconf.pt")
    weights_note = "retrained Aug 2026 after the contamination review; preprocess at 416 (reproduction.md)"
    _warned_conf = False

    def __init__(self, device="cuda", conf_keep=0.8, image_resolution=416, model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, image_resolution=image_resolution,
                         model_id=model_id, **kw)

    @classmethod
    def resolve_weights(cls, override=None):
        if override:
            return override
        for d in checkpoint_dirs():
            hits = sorted(glob.glob(os.path.join(str(d), "vggt_omega_1b_512_reproduce*.pt")))
            if hits:
                return hits[0]
        return super().resolve_weights(override)

    def _load(self):
        import torch
        isolate_third_party(("vggt_omega",), "vggt-omega")
        from vggt_omega.models import VGGTOmega
        self.model_id = self.resolve_weights(self._weights_override)
        model = VGGTOmega()
        state = torch.load(self.model_id, map_location="cpu", weights_only=False)
        sd = state.get("model", state) if isinstance(state, dict) else state
        try:
            model.load_state_dict(sd)
        except RuntimeError as ex:
            res = model.load_state_dict(sd, strict=False)
            print(f"[vggt_omega_repro] strict load failed ({str(ex)[:120]}...); loaded non-strictly: "
                  f"{len(res.missing_keys)} missing keys (e.g. {res.missing_keys[:4]}), "
                  f"{len(res.unexpected_keys)} unexpected (e.g. {res.unexpected_keys[:4]})")
        del state
        self._model = model.to(self.device).eval()
        self._torch = torch

    def _raw(self, image_paths) -> RawViews:
        import torch
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        images = load_and_preprocess_images([str(p) for p in image_paths],
                                            image_resolution=self.image_resolution).to(self.device)
        with torch.inference_mode():
            pred = self._model(images)
        extri, intri = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])

        def np_(x):
            return np.asarray(x.detach().float().cpu().numpy(), dtype=float)

        extri, intri = np_(extri), np_(intri)
        extri = extri[0] if extri.ndim == 4 else extri
        intri = intri[0] if intri.ndim == 4 else intri
        poses = np.tile(np.eye(4), (extri.shape[0], 1, 1))
        poses[:, :3, :4] = extri[:, :3, :4]
        depth = np_(pred["depth"]).squeeze(0)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if pred.get("depth_conf") is not None:
            conf = np_(pred["depth_conf"])
            conf = conf.squeeze(0) if conf.ndim == 4 else conf
        else:
            if not VGGTOmegaReproBackbone._warned_conf:
                print("[vggt_omega_repro] no depth_conf output; keeping every point (conf = 1)")
                VGGTOmegaReproBackbone._warned_conf = True
            conf = np.ones_like(depth)
        rgb = np_(pred["images"])
        rgb = rgb[0] if rgb.ndim == 5 else rgb
        if rgb.shape[1] == 3:
            rgb = rgb.transpose(0, 2, 3, 1)
        return RawViews(poses=poses, rgb=np.clip(rgb, 0, 1), conf=conf, depth=depth, intrinsics=intri)
