"""VGGT-Omega adapter.

API verified against github.com/facebookresearch/vggt-omega README
(2026-08-24):

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    images = load_and_preprocess_images(names, image_resolution=512)
    predictions = model(images)
    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:])
    predictions["depth"], predictions["depth_conf"]

POSE CONVENTION: w2c (VGGT-family camera encoding).

WEIGHTS ARE ACCESS-GATED.  Request access at
huggingface.co/facebook/VGGT-Omega; approval is automated but not instant.
Then either download vggt_omega_1b_512.pt into third_party/checkpoints, or
export HF_TOKEN and let fetch_weights.py pull it.

PAPER NOTE, from their own README: "We recently became aware of an issue
that may have caused benchmark contamination in an ancestor checkpoint of
the released 1B model. As a result, the performance of the released 1B
model as reported in Table 1 and 2 (1B row) may be inflated."  If VGGT-Omega
appears in our tables, that caveat has to be cited next to its row.
"""
from __future__ import annotations

import numpy as np

from .base import register
from .pointmap import PointmapBackbone, RawViews, isolate_third_party


@register("vggt_omega")
class VGGTOmegaBackbone(PointmapBackbone):
    name = "vggt_omega"
    pose_convention = "w2c"
    third_party_paths = ("vggt-omega",)
    requires = ("vggt_omega",)
    weights = "vggt_omega_1b_512.pt"
    weights_file = "vggt_omega_1b_512.pt"
    weights_hub = None
    weights_url = ("https://huggingface.co/facebook/VGGT-Omega/resolve/main/"
                   "vggt_omega_1b_512.pt")
    weights_note = ("ACCESS-GATED: request at huggingface.co/facebook/"
                    "VGGT-Omega, then export HF_TOKEN before fetching")

    def __init__(self, device="cuda", conf_keep=0.8, image_resolution=512,
                 model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, **kw)
        self.image_resolution = image_resolution
        self._weights_override = model_id

    def _load(self):
        import torch
        isolate_third_party(("vggt_omega",), "vggt-omega")
        from vggt_omega.models import VGGTOmega
        self.model_id = self.resolve_weights(self._weights_override)
        model = VGGTOmega()
        state = torch.load(self.model_id, map_location="cpu",
                           weights_only=False)
        model.load_state_dict(state.get("model", state)
                              if isinstance(state, dict) else state)
        del state
        self._model = model.to(self.device).eval()
        self._torch = torch

    def _raw(self, image_paths) -> RawViews:
        import torch
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        images = load_and_preprocess_images(
            [str(p) for p in image_paths],
            image_resolution=self.image_resolution).to(self.device)
        with torch.inference_mode():
            pred = self._model(images)
        extri, intri = encoding_to_camera(pred["pose_enc"],
                                          pred["images"].shape[-2:])

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
        conf = np_(pred["depth_conf"])
        conf = conf.squeeze(0) if conf.ndim == 4 else conf
        rgb = np_(pred["images"])
        rgb = rgb[0] if rgb.ndim == 5 else rgb
        if rgb.shape[1] == 3:
            rgb = rgb.transpose(0, 2, 3, 1)
        return RawViews(poses=poses, rgb=np.clip(rgb, 0, 1), conf=conf,
                        depth=depth, intrinsics=intri)
