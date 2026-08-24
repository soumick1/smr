"""DUSt3R adapter.

API verified against github.com/naver/dust3r (README quick-start, and
dust3r/cloud_opt/base_opt.py) on 2026-08-24:

    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    model  = AsymmetricCroCo3DStereo.from_pretrained(model_id)
    images = load_images(paths, size=512)
    pairs  = make_pairs(images, scene_graph='complete', prefilter=None,
                        symmetrize=True)
    out    = inference(pairs, model, device, batch_size=1)
    scene  = global_aligner(out, device=device, mode=...)
    scene.compute_global_alignment(init="mst", niter=..., schedule=..., lr=...)

POSE CONVENTION: scene.get_im_poses() returns CAMERA-TO-WORLD.  Verified in
the repo: demo.py binds it to `cams2world`, and base_opt.clean_pointcloud
computes `cams = inv(self.get_im_poses())` to obtain world-to-camera.  So we
do NOT invert (same as pi3, opposite of VGGT).

COST NOTE: DUSt3R is pairwise; a K-view scene costs O(K^2) pair inferences
plus a global-alignment optimisation (~300 iters).  This is the slow member
of the family and the reason `scene_graph="swin"` exists for long sequences.
"""
from __future__ import annotations

import numpy as np

from .base import register
from .pointmap import (PointmapBackbone, RawViews,
                       patch_torch_load_legacy_checkpoints)


@register("dust3r")
class DUSt3RBackbone(PointmapBackbone):
    name = "dust3r"
    pose_convention = "c2w"
    third_party_paths = ("dust3r",)
    requires = ("dust3r",)
    # Both routes are official (README "Checkpoints"): the hub integration
    # downloads automatically, or the .pth is served from NAVER Labs.
    weights = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    weights_file = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    weights_url = ("https://download.europe.naverlabs.com/ComputerVision/"
                   "DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    weights_hub = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

    def __init__(self, device="cuda", conf_keep=0.8, image_size=512,
                 niter=300, lr=0.01, schedule="cosine",
                 scene_graph="complete", model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, **kw)
        self.image_size = image_size
        self.niter, self.lr, self.schedule = niter, lr, schedule
        self.scene_graph = scene_graph
        # Lazy on purpose: constructing an adapter must never need
        # a checkpoint on disk (listing metadata, preflight checks
        # and the report script all construct without weights).
        self._weights_override = model_id

    def _load(self):
        patch_torch_load_legacy_checkpoints()
        from dust3r.model import AsymmetricCroCo3DStereo
        # from_pretrained accepts a hub id OR a local .pth path -- the README
        # says so explicitly ("you can put the path to a local checkpoint in
        # model_name if needed").
        self.model_id = self.resolve_weights(self._weights_override)
        self._model = AsymmetricCroCo3DStereo.from_pretrained(
            self.model_id).to(self.device).eval()

    def _raw(self, image_paths) -> RawViews:
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images

        images = load_images(list(image_paths), size=self.image_size)
        pairs = make_pairs(images, scene_graph=self.scene_graph,
                           prefilter=None, symmetrize=True)
        out = inference(pairs, self._model, self.device, batch_size=1)

        pair_only = len(images) == 2
        mode = (GlobalAlignerMode.PairViewer if pair_only
                else GlobalAlignerMode.PointCloudOptimizer)
        scene = global_aligner(out, device=self.device, mode=mode)
        if not pair_only:
            scene.compute_global_alignment(init="mst", niter=self.niter,
                                           schedule=self.schedule, lr=self.lr)

        def np_(x):
            return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach")
                              else x, dtype=float)

        poses = np.stack([np_(p) for p in scene.get_im_poses()])
        depth = np.stack([np_(d) for d in scene.get_depthmaps()])
        conf = np.stack([np_(c) for c in scene.get_conf()])
        intr = np.stack([np_(k) for k in scene.get_intrinsics()])
        rgb = np.stack([np.asarray(im, dtype=float) for im in scene.imgs])
        return RawViews(poses=poses, rgb=rgb, conf=conf, depth=depth,
                        intrinsics=intr,
                        extras=dict(aligner=str(mode), n_pairs=len(pairs)))
