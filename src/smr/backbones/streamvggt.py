"""StreamVGGT adapter (causal streaming VGGT, Zheng et al.).

API verified against github.com/wzzheng/streamvggt demo_gradio.py
(2026-08-24):

    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = StreamVGGT()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    frames = [{"img": images[i].unsqueeze(0)} for i in range(S)]
    output = model.inference(frames)          # NOTE: .inference(), not __call__
    for res in output.ress:                   # per-view dicts
        res['pts3d_in_other_view'], res['conf'],
        res['depth'], res['depth_conf'], res['camera_pose']

The package lives at src/streamvggt inside the repo, so the third_party
path is "streamvggt/src".

POSE CONVENTION: w2c -- `camera_pose` is VGGT's 9-D pose encoding, decoded
by their pose_encoding_to_extri_intri into OpenCV camera-from-world
extrinsics, exactly as in our VGGT adapter.
"""
from __future__ import annotations

import numpy as np

from .base import register
from .pointmap import PointmapBackbone, RawViews, isolate_third_party


@register("streamvggt")
class StreamVGGTBackbone(PointmapBackbone):
    name = "streamvggt"
    pose_convention = "w2c"
    third_party_paths = ("streamvggt/src",)
    requires = ("streamvggt",)
    weights = "StreamVGGT checkpoints.pth"
    weights_file = "streamvggt_checkpoints.pth"
    weights_hub = None
    weights_url = ("https://huggingface.co/lch01/StreamVGGT/resolve/main/"
                   "checkpoints.pth")

    def __init__(self, device="cuda", conf_keep=0.8, model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, **kw)
        self._weights_override = model_id

    def _load(self):
        import torch
        isolate_third_party(("streamvggt", "vggt", "dust3r", "croco"),
                            "streamvggt/src")
        from streamvggt.models.streamvggt import StreamVGGT
        self.model_id = self.resolve_weights(self._weights_override)
        model = StreamVGGT()
        ckpt = torch.load(self.model_id, map_location="cpu",
                          weights_only=False)
        model.load_state_dict(ckpt, strict=True)
        del ckpt
        self._model = model.to(self.device).eval()
        self._torch = torch
        self._dtype = (torch.bfloat16
                       if torch.cuda.is_available() and
                       torch.cuda.get_device_capability()[0] >= 8
                       else torch.float16)

    def _raw(self, image_paths) -> RawViews:
        import torch
        from streamvggt.utils.load_fn import load_and_preprocess_images
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = load_and_preprocess_images(
            [str(p) for p in image_paths]).to(self.device)
        frames = [{"img": images[i].unsqueeze(0)}
                  for i in range(images.shape[0])]
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self._dtype):
                out = self._model.inference(frames)

        def np_(x):
            return np.asarray(x.detach().float().cpu().numpy(), dtype=float)

        depth = np.stack([np_(r["depth"]).squeeze() for r in out.ress])
        conf = np.stack([np_(r["depth_conf"]).squeeze() for r in out.ress])
        pose_enc = torch.stack([r["camera_pose"].squeeze(0)
                                for r in out.ress], 0)[None]
        extri, intri = pose_encoding_to_extri_intri(pose_enc,
                                                    images.shape[-2:])
        extri, intri = np_(extri), np_(intri)
        extri = extri[0] if extri.ndim == 4 else extri
        intri = intri[0] if intri.ndim == 4 else intri
        poses = np.tile(np.eye(4), (extri.shape[0], 1, 1))
        poses[:, :3, :4] = extri[:, :3, :4]           # w2c; socket inverts
        rgb = np_(images).transpose(0, 2, 3, 1)
        return RawViews(poses=poses, rgb=np.clip(rgb, 0, 1), conf=conf,
                        depth=depth, intrinsics=intri)
