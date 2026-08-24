"""Shared machinery for the point-map family of geometry backbones.

Every modern feed-forward geometry model we integrate (DUSt3R, MASt3R,
MUSt3R, CUT3R, Fast3R, StreamR3R, MV-DUSt3R+, VGGT-Omega) emits the same
three things per view:

    * a camera pose            (4, 4)   -- in ITS OWN convention
    * a local point map        (H, W, 3) in camera frame, OR a depth map
    * a confidence map         (H, W)

Everything the scaffold needs -- depth, intrinsics, mask, scene scale -- is
a deterministic function of those.  This module implements that function
ONCE, in pure numpy, so that:

  1. a new backbone adapter only supplies the raw triple (`_raw`) and
     DECLARES its pose convention (`pose_convention`); it never re-derives
     intrinsics or re-invents a normalisation, which is where silent
     convention bugs come from;
  2. the conversion is unit-testable against synthetic cameras with no
     weights, no GPU and no torch (see tests/test_backbones.py), so adding
     a backbone is a small verifiable job rather than an act of faith.

Projection convention is OURS and matches the renderer in smr.scene:
    u = f * X / Z + W / 2 ,   v = f * Y / Z + H / 2
with u, v integer pixel indices and the principal point at the image
centre.  Focals estimated here are therefore directly usable by splat().
"""
from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import Backbone, BackboneOutput

POSE_CONVENTIONS = ("c2w", "w2c")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def ensure_third_party(*subdirs):
    """Put PYTHONPATH-style backbone repos on sys.path.

    DUSt3R and MASt3R ship NO setup.py / pyproject.toml -- only
    requirements.txt -- so `pip install -e` cannot work and their own demos
    simply run from the repo root.  Rather than making every user discover
    that, each adapter declares the third_party subdirectories it needs and
    we insert them here, idempotently, before importing.

    MASt3R additionally vendors dust3r as a submodule at mast3r/dust3r, so
    order matters: a repo's own vendored copy should win over a sibling
    checkout to avoid mixing versions.
    """
    for sub in subdirs:
        d = REPO_ROOT / "third_party" / sub
        if d.is_dir():
            s = str(d)
            if s not in sys.path:
                sys.path.insert(0, s)


# --------------------------------------------------------------- compat ----
# Third-party repos pin ecosystems from their release date.  Rather than
# forcing global downgrades (which would break the other backbones sharing
# this environment), we apply the narrowest possible shims, each documented
# with the upstream cause.

def patch_scipy_hierarchy_distance():
    """MASt3R's sparse aligner calls `scipy.cluster.hierarchy.distance`.

    Older SciPy re-exported `scipy.spatial.distance` as an attribute of
    `scipy.cluster.hierarchy`; current SciPy does not, so `sch.distance`
    raises AttributeError inside sparse_global_alignment's 'hclust-*'
    kinematic mode.  Restoring the alias is exactly the old behaviour --
    `sch.distance.squareform` IS `scipy.spatial.distance.squareform`.

    Returns True if the shim was applied.
    """
    import scipy.cluster.hierarchy as sch
    if hasattr(sch, "distance"):
        return False
    import scipy.spatial.distance as ssd
    sch.distance = ssd
    return True


_ABC_ALIASES = ("Iterable", "Mapping", "MutableMapping", "Sequence",
                "MutableSequence", "Callable", "Hashable", "Set",
                "MutableSet", "Iterator", "Container", "Sized", "Collection",
                "Generator", "Reversible", "ItemsView", "KeysView",
                "ValuesView", "MappingView", "Awaitable", "Coroutine")


def patch_collections_abc_aliases():
    """Restore the `collections.X` aliases that Python 3.10 removed.

    They were deprecated in 3.3 in favour of `collections.abc.X` and deleted
    in 3.10.  Libraries released before that (several in MUSt3R's import
    chain) still use the old spelling, which surfaces as
    `AttributeError: module 'collections' has no attribute 'Iterable'`.
    Re-binding the alias to the identical object in collections.abc restores
    exactly the pre-3.10 behaviour -- it adds names back, changes none.

    Returns the list of names restored (empty on Python < 3.10).
    """
    import collections
    import collections.abc as cabc
    restored = []
    for n in _ABC_ALIASES:
        if not hasattr(collections, n) and hasattr(cabc, n):
            setattr(collections, n, getattr(cabc, n))
            restored.append(n)
    return restored


def center_crop_box(h, w, crop_h, crop_w):
    """(top, left) of a centred crop, matching torchvision's rounding."""
    return int(round((h - crop_h) / 2.0)), int(round((w - crop_w) / 2.0))


def patch_torchvision_center_crop():
    """Teach an old torchvision's center_crop to accept tensors.

    torchvision <= ~0.9 implemented center_crop for PIL only:
        image_width, image_height = img.size
    On a torch.Tensor, `.size` is a METHOD, so this raises
        TypeError: cannot unpack non-iterable builtin_function_or_method
    MUSt3R's loader does CenterCrop(ToTensor(image)), which hits it.

    The real fix is to upgrade torchvision to match the installed torch;
    this shim exists so a backbone can run without an environment change
    mid-deadline.  It delegates to the original for PIL inputs and slices
    tensors directly (exact, and dependency-free).

    Returns True if the shim was installed, False if unnecessary.
    """
    import torchvision.transforms.functional as F
    if getattr(F.center_crop, "__SMR_PATCHED__", False):
        return False
    import numbers
    import torch
    original = F.center_crop
    try:                                   # already tensor-capable? leave it
        probe = torch.zeros(1, 4, 4)
        original(probe, [2, 2])
        return False
    except Exception:
        pass

    def center_crop(img, output_size):
        if torch.is_tensor(img):
            if isinstance(output_size, numbers.Number):
                output_size = (int(output_size), int(output_size))
            elif len(output_size) == 1:
                output_size = (output_size[0], output_size[0])
            ch, cw = int(output_size[0]), int(output_size[1])
            h, w = img.shape[-2], img.shape[-1]
            if ch > h or cw > w:
                raise ValueError(f"crop {(ch, cw)} exceeds image {(h, w)}; "
                                 f"upgrade torchvision for padded crops")
            top, left = center_crop_box(h, w, ch, cw)
            return img[..., top:top + ch, left:left + cw]
        return original(img, output_size)

    center_crop.__SMR_PATCHED__ = True
    F.center_crop = center_crop
    return True


def ensure_module_stub(name, permissive=False, **attrs):
    """Install a minimal stand-in module so an unused import can succeed.

    Used for optional dependencies of code paths we deliberately do not
    exercise.  Never shadows a real installation: if the module imports, we
    leave it alone.  Returns True if a stub was installed.
    """
    import importlib, types
    try:
        importlib.import_module(name)
        return False
    except ImportError:
        pass
    mod = types.ModuleType(name)
    mod.__SMR_STUB__ = True
    for k, v in attrs.items():
        setattr(mod, k, v)
    if permissive:
        # PEP 562 module __getattr__: satisfies `from stub import Anything`
        # by minting a dummy class on demand.  Used for training-only and
        # visualisation-only imports that inference never calls.
        # Dunders MUST raise AttributeError: importlib asks a package for
        # __path__ and would otherwise try to iterate a class.
        def _missing(attr, _n=name):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            return type(attr, (object,), {"__SMR_STUB__": True, "__mod__": _n})
        mod.__getattr__ = _missing
        mod.__path__ = []          # act as a package so submodules resolve
    sys.modules[name] = mod
    if "." in name:                      # bind onto the parent package too,
        parent, _, leaf = name.rpartition(".")   # for `from pkg.sub import X`
        try:
            setattr(importlib.import_module(parent), leaf, mod)
        except Exception:
            pass
    return True


class _StubWeightsEnum:
    """Stand-in for torchvision's *_Weights enums (added in 0.13)."""
    IMAGENET1K_V1 = "IMAGENET1K_V1"
    IMAGENET1K_V2 = "IMAGENET1K_V2"
    DEFAULT = "IMAGENET1K_V1"

    def __init__(self, *a, **k):
        pass


def patch_torchvision_weight_enums():
    """Add the `*_Weights` enums that torchvision <0.13 lacks.

    torchmetrics imports `from torchvision.models import VGG16_Weights` at
    MODULE level (functional/image/dists.py), so on an old torchvision the
    whole Lightning stack fails to import -- which is how fast3r dies.
    These stubs only need to satisfy the import and default arguments of
    metrics we never call.

    The real fix is upgrading torchvision to match torch; this keeps a
    backbone usable until then.  Returns the names added.
    """
    import torchvision.models as tvm
    added = []
    for nm in ("VGG16_Weights", "VGG11_Weights", "VGG19_Weights",
               "AlexNet_Weights", "SqueezeNet1_1_Weights",
               "Inception_V3_Weights", "ResNet18_Weights",
               "ResNet50_Weights"):
        if not hasattr(tvm, nm):
            setattr(tvm, nm, type(nm, (_StubWeightsEnum,), {}))
            added.append(nm)
    return added


def patch_torch_load_legacy_checkpoints():
    """Let torch >= 2.6 load the pre-2.6 checkpoints these repos publish.

    PyTorch 2.6 flipped `torch.load`'s `weights_only` default to True.
    DUSt3R/MASt3R/MV-DUSt3R checkpoints pickle an `argparse.Namespace` of
    training args, so they now fail with
        UnsupportedGlobal: GLOBAL argparse.Namespace was not an allowed global

    We allowlist exactly that class rather than disabling the safety
    default wholesale.  Only ever applied to checkpoints fetched from the
    official URLs recorded on each adapter.  No-op on torch < 2.6.
    """
    import argparse
    import torch
    add = getattr(torch.serialization, "add_safe_globals", None)
    if add is None:                       # torch < 2.6: nothing to do
        return False
    add([argparse.Namespace])
    return True


def isolate_third_party(prefixes, *subdirs):
    """Give this backbone's VENDORED copies priority over cached ones.

    Several repos ship their own fork of dust3r (mast3r/dust3r,
    mvdust3r/dust3r, must3r/dust3r).  In a process that already imported a
    different backbone's copy -- exactly what scripts/check_backbones.py
    does -- `import dust3r` returns the cached wrong module, and a class the
    fork added (e.g. AsymmetricCroCo3DStereoMultiView) appears to be
    missing.  Purge the cached names, then re-prioritise this repo's paths.
    """
    dropped = [n for n in list(sys.modules)
               if any(n == p or n.startswith(p + ".") for p in prefixes)]
    for n in dropped:
        del sys.modules[n]
    for sub in reversed(subdirs):
        d = REPO_ROOT / "third_party" / sub
        if d.is_dir():
            s = str(d)
            if s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)
    return dropped


def checkpoint_dirs():
    """Where local .pth files are looked for, in priority order."""
    dirs = []
    env = os.environ.get("SMR_CKPT_DIR")
    if env:
        dirs.append(pathlib.Path(env))
    dirs += [REPO_ROOT / "third_party" / "checkpoints",
             REPO_ROOT / "checkpoints"]
    return dirs


def find_checkpoint(filename):
    """Return the first existing local checkpoint path, else None."""
    if not filename:
        return None
    for d in checkpoint_dirs():
        p = d / filename
        if p.exists():
            return str(p)
    return None


@dataclass
class RawViews:
    """What a backbone hands us, in its own conventions, before assembly.

    Supply EITHER `pts_local` (camera-frame point map) OR `depth`; supply
    `intrinsics` when the model reports them, otherwise they are estimated
    from the point map.
    """
    poses: np.ndarray                        # (K, 4, 4), convention declared
    rgb: np.ndarray                          # (K, H, W, 3) in [0, 1]
    conf: np.ndarray                         # (K, H, W), higher is better
    pts_local: Optional[np.ndarray] = None   # (K, H, W, 3) camera frame
    depth: Optional[np.ndarray] = None       # (K, H, W)
    intrinsics: Optional[np.ndarray] = None  # (3, 3) or (K, 3, 3)
    features: Optional[np.ndarray] = None    # (K, d) pooled, optional
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------- geometry --
def depth_from_pts_local(pts_local):
    """Camera-frame point map -> depth (the z channel)."""
    return np.ascontiguousarray(np.asarray(pts_local, float)[..., 2])


def pts_local_from_depth(depth, f):
    """Depth -> camera-frame point map under our projection convention."""
    depth = np.asarray(depth, float)
    H, W = depth.shape
    u = np.arange(W)[None, :] - W / 2.0
    v = np.arange(H)[:, None] - H / 2.0
    return np.stack([u / f * depth, v / f * depth, depth], axis=-1)


def focal_from_pts_local(pts, valid=None):
    """Closed-form least-squares focal from one camera-frame point map.

    For every valid pixel our convention gives two linear equations in f:
        (u - W/2) * Z = f * X ,    (v - H/2) * Z = f * Y
    so f = (a . b) / (a . a) with a = [X; Y], b = [(u-cx)Z; (v-cy)Z].
    Exact for a pinhole, no iteration, no optimiser.
    """
    pts = np.asarray(pts, float)
    H, W, _ = pts.shape
    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
    m = np.isfinite(Z) & (Z > 1e-6) & np.isfinite(X) & np.isfinite(Y)
    if valid is not None:
        m = m & np.asarray(valid, bool)
    if int(m.sum()) < 16:                       # degenerate: refuse to guess
        return float(max(H, W))
    u = np.broadcast_to(np.arange(W)[None, :] - W / 2.0, (H, W))[m]
    v = np.broadcast_to(np.arange(H)[:, None] - H / 2.0, (H, W))[m]
    a = np.concatenate([X[m], Y[m]])
    b = np.concatenate([u * Z[m], v * Z[m]])
    den = float(a @ a)
    return float(a @ b / den) if den > 1e-12 else float(max(H, W))


# ---------------------------------------------------------------- assembly --
def assemble(raw: RawViews, pose_convention: str,
             conf_keep: float = 0.8) -> BackboneOutput:
    """RawViews (their conventions) -> BackboneOutput (ours).

    The single place in the codebase where a backbone convention is applied.
    Steps, in order: pose convention -> depth -> confidence mask ->
    intrinsics -> per-scene scale gauge (median confident depth := 1).
    """
    if pose_convention not in POSE_CONVENTIONS:
        raise ValueError(f"pose_convention must be one of {POSE_CONVENTIONS}, "
                         f"got {pose_convention!r}")

    poses = np.array(raw.poses, dtype=float, copy=True)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (K,4,4), got {poses.shape}")
    if pose_convention == "w2c":
        poses = np.linalg.inv(poses)

    rgb = np.clip(np.asarray(raw.rgb, dtype=float), 0.0, 1.0)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must be (K,H,W,3), got {rgb.shape}")
    K_, H, W = rgb.shape[:3]
    if poses.shape[0] != K_:
        raise ValueError(f"{poses.shape[0]} poses for {K_} images")

    if raw.depth is not None:
        depth = np.array(raw.depth, dtype=float, copy=True)
    elif raw.pts_local is not None:
        depth = depth_from_pts_local(raw.pts_local).copy()
    else:
        raise ValueError("RawViews needs either depth or pts_local")
    if depth.shape != (K_, H, W):
        raise ValueError(f"depth must be {(K_, H, W)}, got {depth.shape}")

    conf = np.asarray(raw.conf, dtype=float)
    if conf.shape != (K_, H, W):
        raise ValueError(f"conf must be {(K_, H, W)}, got {conf.shape}")
    finite = np.isfinite(depth) & (depth > 0)
    thr = float(np.quantile(conf[np.isfinite(conf)], 1.0 - conf_keep))
    mask = (conf >= thr) & finite
    if not mask.any():
        raise ValueError("confidence mask is empty; check conf_keep / conf")

    if raw.intrinsics is not None:
        intr = np.asarray(raw.intrinsics, dtype=float)
        intr_all = intr if intr.ndim == 3 else np.tile(intr, (K_, 1, 1))
        intr0 = np.array(intr_all[0], dtype=float)
    else:
        if raw.pts_local is None:
            raise ValueError("intrinsics absent and pts_local absent: cannot "
                             "estimate focal")
        pts = np.asarray(raw.pts_local, float)
        fs = [focal_from_pts_local(pts[i], mask[i]) for i in range(K_)]
        f = float(np.median(fs))
        intr0 = np.array([[f, 0.0, W / 2.0],
                          [0.0, f, H / 2.0],
                          [0.0, 0.0, 1.0]])
        intr_all = np.tile(intr0, (K_, 1, 1))

    s = float(np.median(depth[mask]))
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"degenerate scene scale {s}")
    depth = depth / s
    poses[:, :3, 3] /= s

    extras = dict(raw.extras or {})
    extras.update(conf=conf, scene_scale=s, intrinsics_all=intr_all,
                  pose_convention=pose_convention)
    return BackboneOutput(poses=poses, intrinsics=intr0, depth=depth,
                          rgb=rgb, mask=mask, features=raw.features,
                          extras=extras)


# ------------------------------------------------------------------- base ---
class PointmapBackbone(Backbone):
    """Base for point-map backbones.

    A subclass supplies:
        pose_convention : "c2w" or "w2c"          (DECLARE, never assume)
        requires        : importable module names, checked before load
        weights         : default checkpoint id, recorded in reports
        _load()         : build self._model
        _raw(paths)     : -> RawViews
    and inherits everything else.
    """

    name = "pointmap"
    pose_convention = "c2w"
    requires: tuple = ()
    weights: Optional[str] = None          # human-readable default id
    weights_file: Optional[str] = None     # local .pth filename, if any
    weights_url: Optional[str] = None      # verified download URL
    weights_hub: Optional[str] = None      # HF hub id, if supported
    experimental = False

    # ---------------------------------------------------------- checkpoints
    @classmethod
    def resolve_weights(cls, override=None):
        """Local checkpoint if present, else the hub id, else a loud error
        carrying the exact download command.

        Local files win so that an air-gapped or rate-limited server never
        silently re-downloads, and so the paper can pin a byte-identical
        checkpoint.
        """
        if override:
            return override
        local = find_checkpoint(cls.weights_file)
        if local:
            return local
        if cls.weights_hub:
            return cls.weights_hub
        raise FileNotFoundError(
            f"{cls.name}: checkpoint '{cls.weights_file}' not found in "
            f"{[str(d) for d in checkpoint_dirs()]} and this model has no "
            f"HuggingFace hub id.  Fetch it with:\n"
            f"    python scripts/fetch_weights.py --backbones {cls.name}\n"
            f"or manually:\n"
            f"    mkdir -p third_party/checkpoints && wget "
            f"{cls.weights_url} -P third_party/checkpoints/")

    def __init__(self, device="cuda", conf_keep=0.8, **kw):
        self.device = device
        self.conf_keep = conf_keep
        self._model = None
        self._opts = dict(kw)

    # -------------------------------------------------------------- loading
    def preflight(self):
        """Report which required third-party modules are importable.

        Used by scripts/check_backbones.py to fail early with a useful
        message instead of deep inside an inference call.
        """
        import importlib
        self.bootstrap()
        status = {}
        for mod in self.requires:
            try:
                importlib.import_module(mod)
                status[mod] = True
            except Exception:
                status[mod] = False
        return status

    third_party_paths: tuple = ()          # subdirs of third_party/ to add

    def bootstrap(self):
        """Make this backbone's package importable (idempotent)."""
        if self.third_party_paths:
            ensure_third_party(*self.third_party_paths)

    def missing_repos(self):
        """Declared third_party subdirectories that are not on disk."""
        return [s for s in self.third_party_paths
                if not (REPO_ROOT / "third_party" / s).is_dir()]

    def _require(self):
        missing = [m for m, ok in self.preflight().items() if not ok]
        if not missing:
            return
        # "not cloned" and "cloned but not importable" need different fixes,
        # so say which one this is instead of one generic message.
        repos = self.missing_repos()
        if repos:
            raise ImportError(
                f"{self.name}: repository not cloned -- expected "
                f"third_party/{repos[0]}. Run:\n"
                f"    bash server/setup_backbones.sh {self.name}")
        raise ImportError(
            f"{self.name}: third_party/{self.third_party_paths[0]} exists but "
            f"module(s) {missing} are not importable -- the package is "
            f"probably not installed. Run:\n"
            f"    pip install -e third_party/{self.third_party_paths[0]} "
            f"--no-deps\n"
            f"(--no-deps matters: several of these repos pin numpy<2 and "
            f"would downgrade the environment the other backbones share.)")

    def _load(self):                                   # pragma: no cover
        raise NotImplementedError

    def _raw(self, image_paths) -> RawViews:           # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------ inference
    def infer(self, image_paths) -> BackboneOutput:
        if self._model is None:
            self._require()
            self._load()
        raw = self._raw(list(image_paths))
        return assemble(raw, self.pose_convention, self.conf_keep)
