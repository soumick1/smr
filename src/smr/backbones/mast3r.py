"""MASt3R adapter.

API verified against github.com/naver/mast3r on 2026-08-24:

    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    model = AsymmetricMASt3R.from_pretrained(model_id)
    scene = sparse_global_alignment(img_paths, pairs, cache_path, model, ...)

Scene accessors (mast3r/cloud_opt/sparse_ga.py, verified):
    scene.get_im_poses()        -> self.cam2w              (CAMERA-TO-WORLD)
    scene.get_dense_pts3d(...)  -> (pts3d, depthmaps, confs)
    scene.intrinsics            -> list of (3,3)
    scene.imgs                  -> list of HxWx3
Note the dense depthmaps come back FLAT (one vector per view); we reshape to
the image grid, which is why rgb shapes are read first.

POSE CONVENTION: camera-to-world (the attribute is literally `cam2w`, and
clean_pointcloud calls inv(cam2w) to get the world-to-camera direction).

`aligner="dust3r"` is offered as a fallback: MASt3R's predictions are
DUSt3R-compatible, so the standard global aligner also works and avoids the
sparse-GA cache directory.  Sparse GA is MASt3R's own recipe and is default.
"""
from __future__ import annotations

import tempfile

import numpy as np

from .base import register
from .pointmap import (PointmapBackbone, RawViews,
                       patch_torch_load_legacy_checkpoints,
                       patch_scipy_hierarchy_distance)


@register("mast3r")
class MASt3RBackbone(PointmapBackbone):
    name = "mast3r"
    pose_convention = "c2w"
    third_party_paths = ("mast3r", "mast3r/dust3r", "dust3r")
    requires = ("mast3r", "dust3r")
    weights = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    weights_file = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    weights_url = ("https://download.europe.naverlabs.com/ComputerVision/"
                   "MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_"
                   "metric.pth")
    weights_hub = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # Optional: MASt3R-SfM image retrieval (not needed for our pointmap
    # path).  Both files must sit in the same directory.  README-verified.
    _RB = ("https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
           "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_")
    weights_extras = (
        ("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_"
         "trainingfree.pth", _RB + "trainingfree.pth"),
        ("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_"
         "codebook.pkl", _RB + "codebook.pkl"),
    )

    def __init__(self, device="cuda", conf_keep=0.8, image_size=512,
                 aligner="sparse", subsample=8, cache_dir=None,
                 kinematic_mode="hclust-ward",
                 scene_graph="complete", niter=300, lr=0.01,
                 schedule="cosine", model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, **kw)
        self.image_size, self.aligner = image_size, aligner
        self.subsample, self.cache_dir = subsample, cache_dir
        # 'mst' is their scipy-free alternative if the shim ever fails.
        self.kinematic_mode = kinematic_mode
        self.scene_graph = scene_graph
        self.niter, self.lr, self.schedule = niter, lr, schedule
        # Lazy on purpose: constructing an adapter must never need
        # a checkpoint on disk (listing metadata, preflight checks
        # and the report script all construct without weights).
        self._weights_override = model_id

    def _load(self):
        patch_torch_load_legacy_checkpoints()
        from mast3r.model import AsymmetricMASt3R
        self.model_id = self.resolve_weights(self._weights_override)
        self._model = AsymmetricMASt3R.from_pretrained(
            self.model_id).to(self.device).eval()

    # ------------------------------------------------------------------ raw
    def _raw(self, image_paths) -> RawViews:
        paths = list(image_paths)
        return (self._raw_sparse(paths) if self.aligner == "sparse"
                else self._raw_dust3r(paths))

    def _raw_sparse(self, paths) -> RawViews:
        from dust3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

        # Their default kinematic mode ('hclust-ward') calls
        # scipy.cluster.hierarchy.distance, an alias current SciPy dropped.
        patch_scipy_hierarchy_distance()

        images = load_images(paths, size=self.image_size)
        pairs = make_pairs(images, scene_graph=self.scene_graph,
                           prefilter=None, symmetrize=True)
        def np_(x):
            return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach")
                              else x, dtype=float)

        # sparse_global_alignment writes per-image features and per-pair
        # correspondences into `cache` (hundreds of MB for a 20-image pass).
        # v70: a fresh mkdtemp per call was never removed and filled the
        # disk during the nine-backbone sweep; the cache now lives only for
        # the duration of the call unless the user pins cache_dir.
        with tempfile.TemporaryDirectory(prefix="mast3r_cache_") as tmp:
            cache = self.cache_dir or tmp
            scene = sparse_global_alignment(paths, pairs, cache, self._model,
                                            subsample=self.subsample,
                                            kinematic_mode=self.kinematic_mode,
                                            device=self.device)
            rgb = np.stack([np.asarray(im, dtype=float) for im in scene.imgs])
            K_, H, W = rgb.shape[:3]
            poses = np.stack([np_(p) for p in scene.get_im_poses()])
            intr = np.stack([np_(k) for k in scene.intrinsics])
            _pts, depths, confs = scene.get_dense_pts3d(subsample=self.subsample)
            depth = np.stack([np_(d).reshape(H, W) for d in depths])
            conf = np.stack([np_(c).reshape(H, W) for c in confs])
        return RawViews(poses=poses, rgb=rgb, conf=conf, depth=depth,
                        intrinsics=intr, extras=dict(aligner="sparse_ga"))

    def _raw_dust3r(self, paths) -> RawViews:
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images

        images = load_images(paths, size=self.image_size)
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

        return RawViews(
            poses=np.stack([np_(p) for p in scene.get_im_poses()]),
            rgb=np.stack([np.asarray(im, float) for im in scene.imgs]),
            conf=np.stack([np_(c) for c in scene.get_conf()]),
            depth=np.stack([np_(d) for d in scene.get_depthmaps()]),
            intrinsics=np.stack([np_(k) for k in scene.get_intrinsics()]),
            extras=dict(aligner="dust3r_global"))
