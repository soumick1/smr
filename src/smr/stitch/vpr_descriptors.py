"""Established visual-place-recognition descriptors as SMR retrieval cues (v182).

    from smr.stitch.vpr_descriptors import vpr_descriptors
    D = vpr_descriptors(paths, "eigenplaces", device="cuda")          # (N, 2048), L2-normalised
    D = vpr_descriptors(paths, "dino+eigenplaces", device="cuda")     # concatenation of unit vectors, renormalised
    D = vpr_descriptors(paths, "salad@4096", device="cuda")           # seeded Gaussian projection to 4096 dims

Kinds (each at the input size its authors evaluate at, ImageNet-normalised RGB unless noted):
  eigenplaces  Berton et al., ICCV 2023. torch.hub gmberton/eigenplaces, ResNet50, 2048-D, 512x512.
  cosplace     Berton et al., CVPR 2022. torch.hub gmberton/cosplace, ResNet50, 2048-D, 512x512.
  salad        Izquierdo & Civera, CVPR 2024. torch.hub serizba/salad, DINOv2-B + SALAD, 8448-D, 322x322.
  boq          Ali-bey et al., CVPR 2024. torch.hub amaralibey/bag-of-queries, DINOv2 backbone, 12288-D, 322x322.
  netvlad      Arandjelovic et al., CVPR 2016 (VGG16, Pitts30k, PCA-whitened 4096-D) through hloc
               (pip install git+https://github.com/cvg/Hierarchical-Localization.git); [0,1] RGB, 640x480.
  mixvpr       Ali-bey et al., WACV 2023. Needs the official repo at third_party/MixVPR (git clone
               https://github.com/amaralibey/MixVPR) and the checkpoint
               third_party/checkpoints/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt (gdown 1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L);
               ResNet50 to layer 3, 320x320, 4096-D, built from models.helper (their main.py needs the training
               dataset on disk just to import). Loaded strictly.
  dino         DINOv2 ViT-S/14 CLS, 384-D (the production cue; smr.stitch.passes.dino_descriptors).
  a+b          hybrid: concatenation of the two unit vectors, renormalised (cosine = mean of the two cosines).
  kind@N       seeded Gaussian projection of the final vector to N dimensions (keeps the RLS cue memory, which is
               O(D^2) in the cue dimension, bounded for the 8k/12k-D methods).
Everything is cached by pilot_a under the descriptor name, so each keyframe is encoded once per kind.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPECS = {
    "eigenplaces": dict(hub=("gmberton/eigenplaces", "get_trained_model"), kw=dict(backbone="ResNet50", fc_output_dim=2048),
                        size=(512, 512), normalize=True),
    "cosplace": dict(hub=("gmberton/cosplace", "get_trained_model"), kw=dict(backbone="ResNet50", fc_output_dim=2048),
                     size=(512, 512), normalize=True),
    "salad": dict(hub=("serizba/salad", "dinov2_salad"), kw={}, size=(322, 322), normalize=True),
    "boq": dict(hub=("amaralibey/bag-of-queries", "get_trained_boq"), kw=dict(backbone_name="dinov2", output_dim=12288),
                size=(322, 322), normalize=True),
    "netvlad": dict(hloc=True, size=(640, 480), normalize=False),
    "mixvpr": dict(mixvpr=True, size=(320, 320), normalize=True),
}


def _l2(x):
    x = np.asarray(x, np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def parse_kind(kind):
    """'dino+salad@4096' -> (['dino', 'salad'], 4096)"""
    proj = None
    if "@" in kind:
        kind, p = kind.rsplit("@", 1)
        proj = int(p)
    parts = [k.strip() for k in kind.split("+") if k.strip()]
    for k in parts:
        if k not in SPECS and k != "dino":
            raise ValueError(f"unknown descriptor kind {k!r}; known: dino, {', '.join(SPECS)}; hybrids a+b; projection kind@N")
    return parts, proj


def project(D, dim, seed=0):
    """Seeded Gaussian projection to `dim` dimensions, renormalised (cosines preserved in expectation)."""
    D = np.asarray(D, np.float64)
    if dim is None or dim >= D.shape[1]:
        return _l2(D)
    R = np.random.default_rng(seed).standard_normal((D.shape[1], dim)) / np.sqrt(dim)
    return _l2(D @ R)


def _load_batch(paths, size, normalize, device):
    import torch
    from PIL import Image
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize(size, Image.BICUBIC)
        ims.append(torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1))
    x = torch.stack(ims).to(device)
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x


def _first_tensor(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for key in ("global_descriptor", "descriptor", "desc", "features"):
            if key in out:
                return out[key]
        return next(v for v in out.values() if isinstance(v, torch.Tensor))
    if isinstance(out, (list, tuple)):
        return _first_tensor(out[0])
    raise TypeError(f"unexpected model output {type(out)}")


def _build(kind, device):
    """Return (model, forward) for one kind; forward maps a (B,3,H,W) tensor to (B,D) descriptors."""
    import torch
    spec = SPECS[kind]
    if "hub" in spec:
        repo, entry = spec["hub"]
        model = torch.hub.load(repo, entry, trust_repo=True, **spec["kw"]).to(device).eval()
        return model, lambda x: _first_tensor(model(x))
    if spec.get("hloc"):
        try:
            from hloc.extractors.netvlad import NetVLAD
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError("netvlad needs hloc: pip install git+https://github.com/cvg/Hierarchical-Localization.git "
                               f"(import failed: {ex})") from ex
        model = NetVLAD({"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}).to(device).eval()
        return model, lambda x: _first_tensor(model({"image": x}))
    if spec.get("mixvpr"):
        repo = ROOT / "third_party" / "MixVPR"
        ckpt = ROOT / "third_party" / "checkpoints" / "resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"
        if not repo.exists() or not ckpt.exists():
            raise RuntimeError(f"mixvpr needs the official repo at {repo} (git clone https://github.com/amaralibey/MixVPR) "
                               f"and the checkpoint {ckpt} (gdown 1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L)")
        # main.py imports the GSV-Cities training dataloader, which refuses to import without that dataset on disk;
        # the model itself is backbone + aggregator from models.helper, and the checkpoint keys are named exactly so
        sys.path.insert(0, str(repo))
        from models import helper  # type: ignore  # noqa: E402

        class _MixVPR(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = helper.get_backbone("resnet50", pretrained=False, layers_to_freeze=1, layers_to_crop=[4])
                self.aggregator = helper.get_aggregator("MixVPR", dict(in_channels=1024, in_h=20, in_w=20, out_channels=1024,
                                                                       mix_depth=4, mlp_ratio=1, out_rows=4))

            def forward(self, x):
                return self.aggregator(self.backbone(x))

        model = _MixVPR()
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) and "state_dict" in state else state
        model.load_state_dict(state)                       # strict: the authors' weights or nothing
        model = model.to(device).eval()
        return model, lambda x: _first_tensor(model(x))
    raise ValueError(kind)


def _encode(paths, kind, device, batch):
    import torch
    if kind == "dino":
        from .passes import dino_descriptors
        return _l2(dino_descriptors(paths, device=device))
    spec = SPECS[kind]
    _, fwd = _build(kind, device)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = _load_batch(paths[i:i + batch], spec["size"], spec["normalize"], device)
            f = fwd(x).float().cpu().numpy()
            out.append(f.reshape(f.shape[0], -1))
    return _l2(np.concatenate(out))


def vpr_descriptors(paths, kind, device="cuda", batch=16, seed=0, encoder=None):
    """(N, D) L2-normalised descriptors for `kind` (see module docstring). `encoder` overrides the
    per-kind encoder (testing)."""
    parts, proj = parse_kind(kind)
    enc = encoder or (lambda k: _encode(list(paths), k, device, batch))
    blocks = [np.asarray(enc(k), np.float64) for k in parts]
    D = _l2(np.concatenate([_l2(b) / np.sqrt(len(blocks)) for b in blocks], axis=1))
    return project(D, proj, seed)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="smoke test: encode a few keyframes of one sequence with one or more kinds")
    ap.add_argument("kinds", nargs="+", help="e.g. eigenplaces cosplace salad@4096 netvlad dino")
    ap.add_argument("--gt", default="data/gt/7scenes_chess_seq01.npz", help="GT npz with image_paths")
    ap.add_argument("--frames", default="0,20,500", help="frame indices: adjacent (0,20) should score higher than distant (0,500)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    z = np.load(a.gt, allow_pickle=True)
    paths = [str(x) for x in z["image_paths"]]
    idx = [int(i) for i in a.frames.split(",")]
    sel = [paths[i] for i in idx]
    for kind in a.kinds:
        try:
            D = vpr_descriptors(sel, kind, device=a.device)
            cos = np.round(D @ D.T, 3)
            print(f"{kind:<20} dim {D.shape[1]:>6}  cos({idx[0]},{idx[1]}) {cos[0, 1]:.3f}  cos({idx[0]},{idx[-1]}) {cos[0, -1]:.3f}  ok")
        except Exception as ex:  # noqa: BLE001
            print(f"{kind:<20} FAILED: {str(ex)[:200]}")
