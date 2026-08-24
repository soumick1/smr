"""Fast3R adapter.

API verified against github.com/facebookresearch/fast3r README (2026-08-24):

    from fast3r.models.fast3r import Fast3R
    from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
    from fast3r.dust3r.utils.image import load_images
    from fast3r.dust3r.inference_multiview import inference

    model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
    lit   = MultiViewDUSt3RLitModule.load_for_inference(model)
    images = load_images(filelist, size=512)
    output_dict, _ = inference(images, model, device, dtype=..., verbose=False)
    poses_c2w_batch, focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict['preds'], niter_PnP=100,
        focal_length_estimation_method='first_view_from_global_head')
    camera_poses = poses_c2w_batch[0]          # list of (4,4) CAMERA-TO-WORLD
    preds[i]['pts3d_in_other_view']            # (1,H,W,3) in the GLOBAL frame

POSE CONVENTION: camera-to-world (their README says so explicitly).

NOTE the point maps are global, not per-camera, so we transform them into
each camera's own frame before handing them to the socket -- the socket's
focal estimator and depth both assume camera-frame points.
"""
from __future__ import annotations

import numpy as np

from .base import register
from .pointmap import (PointmapBackbone, RawViews,
                       patch_torchvision_weight_enums,
                       import_with_optional_stubs)


@register("fast3r")
class Fast3RBackbone(PointmapBackbone):
    name = "fast3r"
    pose_convention = "c2w"
    third_party_paths = ("fast3r",)
    requires = ("fast3r",)
    weights = "Fast3R_ViT_Large_512"
    weights_file = None                      # HF-only; nothing to wget
    weights_hub = "jedyang97/Fast3R_ViT_Large_512"
    weights_url = None

    def __init__(self, device="cuda", conf_keep=0.8, image_size=512,
                 niter_pnp=100, model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, **kw)
        self.image_size, self.niter_pnp = image_size, niter_pnp
        self._weights_override = model_id

    # Imported at module level by fast3r but never used at inference:
    #   pl_bolts -> LinearWarmupCosineAnnealingLR, referenced only inside
    #               configure_optimizers (training)
    #   open3d   -> visualisation helpers
    #   wandb    -> logging
    OPTIONAL_AT_INFERENCE = ("pl_bolts", "open3d", "wandb")

    @classmethod
    def _import_lit_module(cls):
        """Import their LightningModule, stubbing training-only deps."""
        def _load():
            from fast3r.models.multiview_dust3r_module import \
                MultiViewDUSt3RLitModule
            return MultiViewDUSt3RLitModule
        return import_with_optional_stubs(_load, cls.OPTIONAL_AT_INFERENCE,
                                          "fast3r")

    def _load(self):
        import torch
        # torchmetrics (pulled in by the Lightning stack) imports
        # torchvision's VGG16_Weights at module level; this torchvision
        # predates that enum.  Stub it before touching fast3r.
        added = patch_torchvision_weight_enums()
        if added:
            print(f"  [note] stubbed torchvision weight enums {added[:2]}... "
                  f"-- upgrade torchvision to match torch to remove this")
        from fast3r.models.fast3r import Fast3R
        MultiViewDUSt3RLitModule = self._import_lit_module()
        self.model_id = self.resolve_weights(self._weights_override)
        self._model = Fast3R.from_pretrained(self.model_id).to(
            self.device).eval()
        self._lit = MultiViewDUSt3RLitModule.load_for_inference(self._model)
        self._lit.eval()
        self._torch = torch

    def _raw(self, image_paths) -> RawViews:
        import torch
        from fast3r.dust3r.inference_multiview import inference
        from fast3r.dust3r.utils.image import load_images
        MultiViewDUSt3RLitModule = self._import_lit_module()

        images = load_images(list(image_paths), size=self.image_size,
                             verbose=False)
        # Their loss_of_one_batch does `autocast(device_type=device.type)`,
        # so a plain "cuda" string fails: pass a real torch.device.
        dev = torch.device(self.device) if isinstance(self.device, str) \
            else self.device
        with torch.no_grad():
            # inference() returns `(result, profiling_info)` ONLY when
            # profiling=True (verified in inference_multiview.py); with
            # profiling=False it returns the collated result dict directly.
            # The README's two-value example passes profiling=True.
            out = inference(images, self._model, dev, dtype=torch.float32,
                            verbose=False, profiling=False)
        if isinstance(out, tuple):                 # profiling build
            out = out[0]
        preds = out["preds"]
        poses_batch, focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
            preds, niter_PnP=self.niter_pnp,
            focal_length_estimation_method="first_view_from_global_head")
        poses = np.stack([np.asarray(p, dtype=float)
                          for p in poses_batch[0]])          # c2w

        def np_(x):
            return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach")
                              else x, dtype=float)

        pts_w, confs, rgbs = [], [], []
        for i, pr in enumerate(preds):
            p = np_(pr["pts3d_in_other_view"])
            pts_w.append(p[0] if p.ndim == 4 else p)
            c = np_(pr["conf"])
            confs.append(c[0] if c.ndim == 3 else c)
            im = np_(images[i]["img"])
            im = im[0] if im.ndim == 4 else im
            if im.shape[0] == 3:                     # CHW -> HWC
                im = im.transpose(1, 2, 0)
            rgbs.append((im + 1.0) / 2.0 if im.min() < -0.01 else im)
        pts_w = np.stack(pts_w)

        # global -> per-camera frame, so depth = z and the focal estimate
        # in the socket is meaningful
        pts_local = np.empty_like(pts_w)
        for i in range(len(poses)):
            Rt = np.linalg.inv(poses[i])
            pts_local[i] = pts_w[i] @ Rt[:3, :3].T + Rt[:3, 3]

        return RawViews(poses=poses, rgb=np.stack(rgbs),
                        conf=np.stack(confs), pts_local=pts_local,
                        extras=dict(focals=np.asarray(focals).tolist()))
