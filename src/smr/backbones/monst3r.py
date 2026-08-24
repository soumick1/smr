"""MonST3R adapter (dynamic-scene DUSt3R, Zhang et al.).

MonST3R vendors its own dust3r fork and keeps DUSt3R's public API, so the
flow is identical to our dust3r adapter -- load AsymmetricCroCo3DStereo
from their checkpoint, pair up, infer, globally align (verified in their
demo.py, which uses GlobalAlignerMode.PointCloudOptimizer with extra
dynamic-scene options we leave at their defaults).

Weights: huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt

POSE CONVENTION: c2w (inherited from DUSt3R's get_im_poses -> cams2world).

WHY IT MATTERS FOR US: it is the dynamic-scene member of the family, so it
is the natural backbone for any sequence where things move -- a regime the
static-scene models degrade in and a fair place to show that a persistent
scaffold does not make that worse.
"""
from __future__ import annotations

import numpy as np

from .base import register
from .pointmap import (PointmapBackbone, RawViews,
                       import_with_optional_stubs, isolate_third_party,
                       patch_torch_load_legacy_checkpoints)


@register("monst3r")
class MonST3RBackbone(PointmapBackbone):
    name = "monst3r"
    pose_convention = "c2w"
    third_party_paths = ("monst3r",)
    requires = ("dust3r",)
    weights = "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
    weights_file = "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
    weights_hub = None
    weights_url = None            # their HF repo exists but the file name
                                  # inside it is not the repo name (my first
                                  # guess 404'd); their own
                                  # data/download_ckpt.sh uses Google Drive:
    weights_gdrive = ("https://drive.google.com/file/d/"
                      "1Z1jO_JmfZj0z3bgMvCwqfUhyZ1bIbc9E/view")
    weights_note = ("Google Drive (per their data/download_ckpt.sh): "
                    "pip install gdown, then fetch_weights.py handles it")

    # imported at module level by their fork, unused at inference:
    #   evo      -- trajectory evaluation (ATE/RPE) in dust3r/utils/vo_eval
    #   seaborn  -- plotting, imported at module level by cloud_opt/
    #               init_im_poses.py (which IS on the inference path, but
    #               only uses seaborn for figures)
    #   open3d, wandb, matplotlib -- visualisation / logging
    #   sam2     -- Segment Anything 2, imported at module level by
    #               cloud_opt/optimizer.py but only CALLED from
    #               refine_motion_mask_w_sam2(), which we switch off below
    OPTIONAL_AT_INFERENCE = ("evo", "seaborn", "sam2", "open3d", "wandb")

    def __init__(self, device="cuda", conf_keep=0.8, image_size=512,
                 niter=300, lr=0.01, schedule="cosine",
                 scene_graph="complete", sam2_mask_refine=False,
                 model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, **kw)
        self.image_size = image_size
        self.niter, self.lr, self.schedule = niter, lr, schedule
        self.scene_graph = scene_graph
        # MonST3R refines dynamic masks with SAM2 by default.  We only need
        # poses + depth, and SAM2 is a separate install with its own
        # checkpoint, so it is off unless you ask for it.
        self.sam2_mask_refine = sam2_mask_refine
        self._weights_override = model_id

    def _load(self):
        # MonST3R ships its OWN dust3r fork; a plain dust3r already imported
        # in this process would shadow it (see mvdust3r).
        isolate_third_party(("dust3r", "croco"), "monst3r")
        patch_torch_load_legacy_checkpoints()
        # dust3r/utils/vo_eval.py imports `evo` (trajectory evaluation) at
        # module level; nothing on the inference path uses it.
        def _model_cls():
            from dust3r.model import AsymmetricCroCo3DStereo
            return AsymmetricCroCo3DStereo
        AsymmetricCroCo3DStereo = import_with_optional_stubs(
            _model_cls, self.OPTIONAL_AT_INFERENCE, "monst3r")
        self.model_id = self.resolve_weights(self._weights_override)
        self._model = AsymmetricCroCo3DStereo.from_pretrained(
            self.model_id).to(self.device).eval()

    def _raw(self, image_paths) -> RawViews:
        isolate_third_party(("dust3r", "croco"), "monst3r")

        def _bits():
            from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
            from dust3r.image_pairs import make_pairs
            from dust3r.inference import inference
            from dust3r.utils.image import load_images
            return (GlobalAlignerMode, global_aligner, make_pairs, inference,
                    load_images)
        (GlobalAlignerMode, global_aligner, make_pairs, inference,
         load_images) = import_with_optional_stubs(
            _bits, self.OPTIONAL_AT_INFERENCE, "monst3r")

        images = load_images(list(image_paths), size=self.image_size)
        pairs = make_pairs(images, scene_graph=self.scene_graph,
                           prefilter=None, symmetrize=True)
        out = inference(pairs, self._model, self.device, batch_size=1)
        pair_only = len(images) == 2
        mode = (GlobalAlignerMode.PairViewer if pair_only
                else GlobalAlignerMode.PointCloudOptimizer)
        kw = dict(verbose=False)
        if not pair_only:
            kw["sam2_mask_refine"] = self.sam2_mask_refine
        scene = global_aligner(out, device=self.device, mode=mode, **kw)
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
            extras=dict(dynamic=True))
