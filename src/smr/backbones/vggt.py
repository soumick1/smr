"""VGGT adapter (server-side, GPU).

Maps the frozen VGGT model onto BackboneOutput.  All downstream code
(scaffold, memory, pipeline) is untouched -- that is the portability claim.

VERIFY-ON-SERVER: the four imports and the prediction keys below follow the
official README of github.com/facebookresearch/vggt at the time of writing;
confirm the names against your clone in third_party/vggt before first run.

Conventions handled here (ours vs theirs):
  * VGGT extrinsics are CAMERA-FROM-WORLD (OpenCV: x_c = R x_w + t,
    x right / y down / z forward -- the same axis convention as our
    renderer).  BackboneOutput.poses are WORLD-FROM-CAMERA, so we invert.
  * Scale: monocular-style outputs are up to scale.  We normalise by the
    median confident depth over all views (per-scene median depth := 1,
    plan Sec. Portability) and scale camera centres identically.
"""
from __future__ import annotations

import numpy as np

from .base import Backbone, BackboneOutput, register


@register("vggt")
class VGGTBackbone(Backbone):
    name = "vggt"

    def __init__(self, device="cuda", model_id="facebook/VGGT-1B",
                 conf_keep=0.8):
        self.device, self.model_id = device, model_id
        self.conf_keep = conf_keep          # keep this fraction by confidence
        self._model = None

    # ------------------------------------------------------------------ load
    def _load(self):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("torch missing: run setup_env.sh "
                              "--with-backbones on the GPU server.") from e
        try:
            from vggt.models.vggt import VGGT          # VERIFY-ON-SERVER
        except ImportError as e:
            raise ImportError(
                "vggt not importable: run server/setup_backbones.sh, then "
                "`pip install -e third_party/vggt` (or add it to "
                "PYTHONPATH).") from e
        import torch
        self._torch = torch
        self._model = VGGT.from_pretrained(self.model_id)  # VERIFY-ON-SERVER
        self._model = self._model.to(self.device).eval()
        self._dtype = (torch.bfloat16 if torch.cuda.is_available() and
                       torch.cuda.get_device_capability()[0] >= 8
                       else torch.float16)

    # ----------------------------------------------------------------- infer
    def infer(self, image_paths) -> BackboneOutput:
        """image_paths: list of image file paths (one scene, K views)."""
        if self._model is None:
            self._load()
        torch = self._torch
        from vggt.utils.load_fn import load_and_preprocess_images  # VERIFY
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # VERIFY

        images = load_and_preprocess_images(list(image_paths)).to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self._dtype):
                pred = self._model(images)             # VERIFY keys below
        # (B, S, ...) -> squeeze batch
        extri, intri = pose_encoding_to_extri_intri(
            pred["pose_enc"], images.shape[-2:])       # VERIFY-ON-SERVER
        extri = extri.squeeze(0).float().cpu().numpy()     # (S, 3, 4) c-f-w
        intri = intri.squeeze(0).float().cpu().numpy()     # (S, 3, 3)
        depth = pred["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()
        conf = pred["depth_conf"].squeeze(0).float().cpu().numpy()
        rgb = images.squeeze(0).permute(0, 2, 3, 1).float().cpu().numpy()

        # camera-from-world -> world-from-camera
        K_views = extri.shape[0]
        poses = np.tile(np.eye(4), (K_views, 1, 1))
        for i in range(K_views):
            R, t = extri[i, :, :3], extri[i, :, 3]
            poses[i, :3, :3] = R.T
            poses[i, :3, 3] = -R.T @ t

        # confidence mask (percentile) and per-scene scale normalisation
        thr = np.quantile(conf, 1.0 - self.conf_keep)
        mask = conf >= thr
        s = float(np.median(depth[mask & (depth > 0)]))
        depth = depth / s
        poses[:, :3, 3] /= s

        return BackboneOutput(poses=poses, intrinsics=intri[0],
                              depth=depth, rgb=np.clip(rgb, 0, 1),
                              mask=mask,
                              extras=dict(conf=conf, scene_scale=s,
                                          intrinsics_all=intri))
