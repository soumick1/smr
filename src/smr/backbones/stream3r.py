"""STream3R adapter (causal / streaming VGGT derivative).

API verified against github.com/NIRVANALAN/STream3R README (2026-08-24):

    from stream3r.models.stream3r import STream3R
    from stream3r.models.components.utils.load_fn import \
        load_and_preprocess_images
    model = STream3R.from_pretrained("yslan/STream3R").to(device)
    images = load_and_preprocess_images(image_names).to(device)
    predictions = model(images, mode="causal")   # or "window" / "full"

Prediction keys (from the model's own docstring): pose_enc [B,S,9],
depth, depth_conf [B,S,H,W], world_points, world_points_conf.

POSE CONVENTION: w2c.  STream3R inherits VGGT's camera head, whose pose
encoding decodes to OpenCV camera-FROM-world extrinsics -- the same
convention our VGGT adapter inverts.  `mode="causal"` is the streaming
setting the paper is about and is our default; "full" reproduces VGGT-like
bidirectional attention.
"""
from __future__ import annotations

import numpy as np

from .base import register
from .pointmap import PointmapBackbone, RawViews


@register("stream3r")
class STream3RBackbone(PointmapBackbone):
    name = "stream3r"
    pose_convention = "w2c"
    third_party_paths = ("STream3R",)
    requires = ("stream3r",)
    weights = "STream3R"
    weights_file = None
    weights_hub = "yslan/STream3R"
    weights_url = None

    def __init__(self, device="cuda", conf_keep=0.8, mode="causal",
                 model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, **kw)
        self.mode = mode
        self._weights_override = model_id

    def _load(self):
        import torch
        from stream3r.models.stream3r import STream3R
        self.model_id = self.resolve_weights(self._weights_override)
        self._model = STream3R.from_pretrained(self.model_id).to(
            self.device).eval()
        self._torch = torch
        self._dtype = (torch.bfloat16
                       if torch.cuda.is_available() and
                       torch.cuda.get_device_capability()[0] >= 8
                       else torch.float16)

    def _decode_pose_enc(self, pose_enc, hw):
        """pose encoding -> (extrinsics w2c, intrinsics).

        STream3R reuses VGGT's camera head; try their util first, then
        VGGT's own, and fail loudly rather than inventing a decoding.
        """
        try:
            from stream3r.models.components.utils.pose_enc import \
                pose_encoding_to_extri_intri
        except ImportError:
            try:
                from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            except ImportError as e:
                raise ImportError(
                    "stream3r: no pose-encoding decoder found in either "
                    "stream3r.models.components.utils.pose_enc or "
                    "vggt.utils.pose_enc; locate it in your checkout and "
                    "update _decode_pose_enc.") from e
        return pose_encoding_to_extri_intri(pose_enc, hw)

    def _raw(self, image_paths) -> RawViews:
        import torch
        from stream3r.models.components.utils.load_fn import \
            load_and_preprocess_images

        images = load_and_preprocess_images(
            [str(p) for p in image_paths]).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self._dtype):
                pred = self._model(images[None], mode=self.mode)

        hw = images.shape[-2:]
        extri, intri = self._decode_pose_enc(pred["pose_enc"], hw)

        def np_(x):
            return np.asarray(x.detach().float().cpu().numpy()
                              if hasattr(x, "detach") else x, dtype=float)

        extri = np_(extri)[0] if np_(extri).ndim == 4 else np_(extri)
        poses = np.tile(np.eye(4), (extri.shape[0], 1, 1))
        poses[:, :3, :4] = extri[:, :3, :4]              # w2c, socket inverts
        intr = np_(intri)
        intr = intr[0] if intr.ndim == 4 else intr

        depth = np_(pred["depth"]).squeeze(0)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        conf = np_(pred.get("depth_conf", pred.get("world_points_conf")))
        conf = conf.squeeze(0) if conf.ndim == 4 else conf
        rgb = np_(images).transpose(0, 2, 3, 1)
        return RawViews(poses=poses, rgb=np.clip(rgb, 0, 1), conf=conf,
                        depth=depth, intrinsics=intr,
                        extras=dict(mode=self.mode))
