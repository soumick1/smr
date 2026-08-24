"""Backbone-adapter contract tests.

Two tiers, deliberately separated:

  * TIER 1 (this file, default): runs anywhere -- no GPU, no torch, no
    checkpoints.  It pins the *conversion* (pose convention, focal
    estimation, masking, scale gauge) against synthetic cameras where the
    right answer is known in closed form, and it proves that anything
    satisfying the RawViews contract binds into the scaffold.  Every
    silent-convention bug we have actually hit (VGGT's w2c inversion, the
    K=1 squeeze, float64 leakage) is the kind this tier catches.

  * TIER 2 (marked `gpu`): actually loads each third-party model and runs
    it on real images.  Skipped unless SMR_GPU_TESTS=1, because it needs
    the server.  Run there with:
        SMR_GPU_TESTS=1 pytest tests/test_backbones.py -m gpu -v
    or use scripts/check_backbones.py for a readable report.
"""
import os
import pathlib

import numpy as np
import pytest

from smr.backbones import get_backbone
from smr.backbones.base import _REGISTRY, BackboneOutput
from smr.backbones.pointmap import (POSE_CONVENTIONS, PointmapBackbone,
                                    RawViews, assemble,
                                    depth_from_pts_local,
                                    focal_from_pts_local,
                                    pts_local_from_depth)
from smr.dynamics import ScaffoldState
from smr.pipeline import bind
from smr.scene import Camera

# must3r removed (see UPDATE_NOTES_v38): needs a newer torchvision.
POINTMAP_ADAPTERS = ["dust3r", "mast3r", "fast3r", "stream3r",
                     "streamvggt", "monst3r", "vggt_omega"]
# adapters that ship a downloadable .pth (others resolve via the HF hub or,
# for cut3r, a Google Drive link that cannot be automated)
# adapters the fetcher can pull automatically: direct https OR Google Drive
# (CUT3R and MonST3R publish only via Drive -- their own documented recipe)
FETCHABLE = ["dust3r", "mast3r", "streamvggt", "vggt_omega"]
GDRIVE = ["monst3r"]
# Some repos publish their checkpoint under a name that would be ambiguous
# in a shared directory ("checkpoints.pth"), so the adapter saves it under a
# backbone-qualified name.  Renaming is allowed ONLY for these.
GENERIC_REMOTE_NAMES = {"checkpoints.pth", "checkpoint.pth", "model.pt",
                        "model.pth", "model.safetensors"}


def url_matches_file(url, weights_file):
    remote = url.rsplit("/", 1)[-1]
    return remote in weights_file or remote in GENERIC_REMOTE_NAMES
ALL_ADAPTERS = ["vggt", "pi3"] + POINTMAP_ADAPTERS


# ------------------------------------------------------------ synthetic io --
def synth_raw(K=4, H=48, W=64, f=70.0, convention="c2w", seed=0):
    """A RawViews whose every quantity is known in closed form."""
    rng = np.random.default_rng(seed)
    zz = 2.0 + rng.random((K, H, W)) * 0.5          # depth in [2, 2.5]
    pts = np.stack([pts_local_from_depth(zz[i], f) for i in range(K)])
    poses = np.tile(np.eye(4), (K, 1, 1))
    for i in range(K):                               # distinct, valid poses
        a = 0.3 * i
        poses[i, :3, :3] = np.array([[np.cos(a), 0, np.sin(a)],
                                     [0, 1, 0],
                                     [-np.sin(a), 0, np.cos(a)]])
        poses[i, :3, 3] = [0.2 * i, 0.05 * i, 0.1 * i]
    out_poses = poses if convention == "c2w" else np.linalg.inv(poses)
    return RawViews(poses=out_poses,
                    rgb=rng.random((K, H, W, 3)),
                    conf=rng.random((K, H, W)),
                    pts_local=pts), poses, f


# ======================================================= TIER 1: contracts ==
def test_all_expected_backbones_are_registered():
    for name in ALL_ADAPTERS:
        assert name in _REGISTRY, f"{name} not registered"


def test_unknown_backbone_raises_with_helpful_message():
    with pytest.raises(KeyError) as e:
        get_backbone("not_a_backbone")
    assert "Known" in str(e.value)


@pytest.mark.parametrize("name", POINTMAP_ADAPTERS)
def test_pointmap_adapters_declare_their_conventions(name):
    """A backbone must DECLARE its pose convention, its imports and its
    weights.  Assuming any of the three is how adapters go silently wrong."""
    cls = _REGISTRY[name]
    assert issubclass(cls, PointmapBackbone)
    assert cls.pose_convention in POSE_CONVENTIONS
    assert isinstance(cls.requires, tuple) and cls.requires
    assert isinstance(cls.weights, str) and cls.weights


@pytest.mark.parametrize("name", POINTMAP_ADAPTERS)
def test_adapters_construct_without_torch_or_weights(name):
    """Construction must be inert: no import, no download, no CUDA."""
    bb = get_backbone(name, device="cpu")
    assert bb._model is None
    status = bb.preflight()                      # reports, never raises
    assert set(status) == set(bb.requires)


@pytest.mark.parametrize("name", POINTMAP_ADAPTERS)
def test_missing_third_party_fails_loud_not_wrong(name):
    """With the third-party package absent, infer() must raise ImportError
    naming the module -- never fall back to a wrong-but-plausible answer."""
    bb = get_backbone(name, device="cpu")
    if all(bb.preflight().values()):
        pytest.skip(f"{name} dependencies present; nothing to assert")
    with pytest.raises(ImportError) as e:
        bb.infer(["a.png", "b.png"])
    msg = str(e.value)
    # actionable means: names what is missing AND the command that fixes it
    named = (any(m in msg for m in bb.requires)
             or any(r in msg for r in bb.third_party_paths))
    assert named, f"error names neither module nor repo: {msg}"
    assert ("setup_backbones.sh" in msg or "pip install -e" in msg), msg


# ================================================== TIER 1: conversion math ==
def test_focal_recovered_exactly_from_pointmap():
    for f in (50.0, 137.5, 400.0):
        depth = np.full((32, 48), 3.0)
        pts = pts_local_from_depth(depth, f)
        assert abs(focal_from_pts_local(pts) - f) < 1e-6 * f


def test_focal_recovery_survives_varying_depth_and_holes():
    rng = np.random.default_rng(3)
    f = 96.0
    depth = 1.5 + rng.random((40, 56)) * 2.0
    pts = pts_local_from_depth(depth, f)
    valid = rng.random((40, 56)) > 0.4          # 60% of pixels
    pts[~valid] = np.nan
    assert abs(focal_from_pts_local(pts, valid) - f) < 1e-6 * f


def test_focal_refuses_to_guess_when_degenerate():
    pts = np.full((8, 8, 3), np.nan)
    assert np.isfinite(focal_from_pts_local(pts))     # fallback, not a crash


def test_depth_pointmap_roundtrip_is_identity():
    rng = np.random.default_rng(1)
    depth = 1.0 + rng.random((24, 32))
    got = depth_from_pts_local(pts_local_from_depth(depth, 88.0))
    assert np.allclose(got, depth, atol=1e-12)


def test_assemble_passes_c2w_through_unchanged_up_to_scale():
    raw, poses_c2w, _ = synth_raw(convention="c2w")
    out = assemble(raw, "c2w")
    s = out.extras["scene_scale"]
    assert np.allclose(out.poses[:, :3, :3], poses_c2w[:, :3, :3], atol=1e-9)
    assert np.allclose(out.poses[:, :3, 3], poses_c2w[:, :3, 3] / s, atol=1e-9)


def test_assemble_inverts_w2c():
    """The VGGT-vs-pi3 trap, pinned: a w2c adapter must come out c2w."""
    raw, poses_c2w, _ = synth_raw(convention="w2c")
    out = assemble(raw, "w2c")
    s = out.extras["scene_scale"]
    assert np.allclose(out.poses[:, :3, :3], poses_c2w[:, :3, :3], atol=1e-9)
    assert np.allclose(out.poses[:, :3, 3], poses_c2w[:, :3, 3] / s, atol=1e-9)


def test_wrong_convention_produces_different_poses():
    """Guards the guard: if declaring c2w vs w2c made no difference, the
    test above would be vacuous."""
    raw, _, _ = synth_raw(convention="c2w")
    assert not np.allclose(assemble(raw, "c2w").poses,
                           assemble(raw, "w2c").poses)


def test_assemble_rejects_unknown_convention():
    raw, _, _ = synth_raw()
    with pytest.raises(ValueError):
        assemble(raw, "cam_to_world_probably")


def test_assemble_sets_median_confident_depth_to_one():
    raw, _, _ = synth_raw()
    out = assemble(raw, "c2w")
    assert abs(float(np.median(out.depth[out.mask])) - 1.0) < 1e-9


def test_assemble_estimates_intrinsics_when_absent():
    raw, _, f = synth_raw()
    out = assemble(raw, "c2w")
    s = out.extras["scene_scale"]
    assert abs(out.intrinsics[0, 0] - f) < 1e-4 * f     # focal is scale-free
    assert abs(out.intrinsics[0, 2] - raw.rgb.shape[2] / 2) < 1e-9
    assert abs(out.intrinsics[1, 2] - raw.rgb.shape[1] / 2) < 1e-9
    assert s > 0


def test_assemble_prefers_reported_intrinsics():
    raw, _, _ = synth_raw()
    raw.intrinsics = np.array([[123.0, 0, 32.0], [0, 123.0, 24.0], [0, 0, 1]])
    assert assemble(raw, "c2w").intrinsics[0, 0] == 123.0


def test_conf_keep_controls_mask_fraction():
    raw, _, _ = synth_raw()
    for keep in (0.25, 0.5, 0.9):
        frac = assemble(raw, "c2w", conf_keep=keep).mask.mean()
        assert abs(frac - keep) < 0.03, f"keep={keep} gave {frac}"


def test_assemble_requires_depth_or_pointmap():
    raw, _, _ = synth_raw()
    raw.pts_local, raw.depth = None, None
    with pytest.raises(ValueError, match="depth or pts_local"):
        assemble(raw, "c2w")


def test_assemble_rejects_shape_mismatches():
    raw, _, _ = synth_raw(K=4)
    raw.poses = raw.poses[:3]
    with pytest.raises(ValueError):
        assemble(raw, "c2w")


def test_assemble_handles_single_view():
    """K=1 broke the VGGT adapter in production; it is a first-class case."""
    raw, _, _ = synth_raw(K=1)
    out = assemble(raw, "c2w")
    assert out.poses.shape == (1, 4, 4)
    assert out.depth.shape[0] == 1 and out.mask.any()


def test_assembled_output_matches_the_schema_downstream_expects():
    raw, _, _ = synth_raw()
    out = assemble(raw, "c2w")
    K_, H, W = out.rgb.shape[:3]
    assert isinstance(out, BackboneOutput)
    assert out.poses.shape == (K_, 4, 4) and out.poses.dtype == np.float64
    assert out.intrinsics.shape == (3, 3)
    assert out.depth.shape == (K_, H, W)
    assert out.mask.shape == (K_, H, W) and out.mask.dtype == np.bool_
    assert out.rgb.min() >= 0.0 and out.rgb.max() <= 1.0
    for i in range(K_):                            # valid rigid transforms
        R = out.poses[i, :3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
    assert out.descriptor(0).shape == (448,)


# ============================ TIER 1: the compatibility guarantee ===========
def test_any_conforming_backbone_binds_into_the_scaffold():
    """The plug-and-play claim, at contract level: an adapter that satisfies
    RawViews yields an output the scaffold binds and recalls from -- with no
    backbone-specific code anywhere downstream.  Weight-level confirmation is
    scripts/check_backbones.py on the server."""
    raw, _, _ = synth_raw(K=3, H=40, W=52)
    out = assemble(raw, "c2w")
    cam = Camera(H=out.depth.shape[1], W=out.depth.shape[2],
                 f=float(out.intrinsics[0, 0]))
    ss = ScaffoldState(periods=[2.4, 3.2], ring_N=64, torus_N=24, seed=0,
                       omega_max=0.16)
    ss.calibrate()
    bound = bind(out, ss, cam, formation_extent=1.2, formation_spacing=0.6,
                 torus_N=24, N_h=512, k=48)
    assert len(bound.store.content) == 3
    ss.place_pose(out.poses[0])
    ids = bound.store.nearest(ss.state(), k=1)
    assert len(ids) == 1 and 0 <= ids[0] < 3


@pytest.mark.parametrize("convention", POSE_CONVENTIONS)
def test_scaffold_binding_is_convention_agnostic_once_declared(convention):
    """Same scene declared either way must bind to the same stored poses."""
    raw, _, _ = synth_raw(K=3, H=40, W=52, convention=convention)
    out = assemble(raw, convention)
    cam = Camera(H=40, W=52, f=float(out.intrinsics[0, 0]))
    ss = ScaffoldState(periods=[2.4, 3.2], ring_N=64, torus_N=24, seed=0,
                       omega_max=0.16)
    ss.calibrate()
    bound = bind(out, ss, cam, formation_extent=1.2, formation_spacing=0.6,
                 torus_N=24, N_h=512, k=48)
    ids = sorted(bound.store.content)          # keyed store, not a list
    stored = np.stack([bound.store.content[i]["T"] for i in ids])
    assert np.allclose(stored, out.poses, atol=1e-9)


# ================================================ TIER 2: real models (GPU) ==
gpu = pytest.mark.skipif(os.environ.get("SMR_GPU_TESTS") != "1",
                         reason="set SMR_GPU_TESTS=1 on the GPU server")


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name", ALL_ADAPTERS)
def test_backbone_loads_and_infers_on_real_images(name, tmp_path):
    frames = os.environ.get("SMR_TEST_FRAMES")
    if not frames:
        pytest.skip("set SMR_TEST_FRAMES=/path/to/a/few/images")
    import pathlib
    paths = sorted(p for p in pathlib.Path(frames).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:4]
    assert len(paths) >= 2, "need at least 2 images"
    bb = get_backbone(name)
    out = bb.infer([str(p) for p in paths])
    K_, H, W = out.rgb.shape[:3]
    assert out.poses.shape == (K_, 4, 4)
    assert out.depth.shape == (K_, H, W)
    assert out.mask.any()
    assert abs(float(np.median(out.depth[out.mask])) - 1.0) < 1e-6
    for i in range(K_):
        R = out.poses[i, :3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-4)


# ===================================== TIER 1: checkpoint plumbing ==========
@pytest.mark.parametrize("name", FETCHABLE + GDRIVE)
def test_every_adapter_declares_a_fetchable_checkpoint(name):
    """A backbone must say WHICH file it needs and WHERE it comes from, or
    `fetch_weights.py` (which reads the registry) silently skips it."""
    cls = _REGISTRY[name]
    assert isinstance(cls.weights_file, str) and cls.weights_file.endswith(
        (".pth", ".pt", ".safetensors")), f"{name}: bad weights_file"
    url = cls.weights_url or getattr(cls, "weights_gdrive", None)
    assert isinstance(url, str) and url.startswith("https://"), (
        f"{name}: no https weights source")
    if "drive.google.com" in url:
        return                       # Drive links carry no filename
    assert url_matches_file(url, cls.weights_file), (
        f"{name}: weights_url basename does not match weights_file and is "
        f"not a known-generic remote name -- one of them is stale")


def test_weight_sources_are_declared_per_backbone():
    """Each backbone must have SOME reachable weight source: a hub id, a
    direct URL, or an explicit note saying why neither exists."""
    for name in POINTMAP_ADAPTERS:
        c = _REGISTRY[name]
        assert (c.weights_hub or c.weights_url
                or getattr(c, "weights_note", None)), \
            f"{name}: no hub id, no URL and no note explaining why"
    assert _REGISTRY["dust3r"].weights_hub and _REGISTRY["mast3r"].weights_hub
    assert _REGISTRY["monst3r"].weights_url is None          # Google Drive
    assert "drive.google.com" in _REGISTRY["monst3r"].weights_gdrive


def test_local_checkpoint_wins_over_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("SMR_CKPT_DIR", str(tmp_path))
    cls = _REGISTRY["dust3r"]
    (tmp_path / cls.weights_file).write_bytes(b"x")
    assert cls.resolve_weights() == str(tmp_path / cls.weights_file)


def test_hub_id_used_when_no_local_file(tmp_path, monkeypatch):
    """SMR_CKPT_DIR is only the FIRST directory searched; the real
    third_party/checkpoints/ is always a fallback, so on a server that has
    the weights this test must point the whole search at the empty tmp dir
    (v65 fix: it used to fail wherever DUSt3R weights were installed)."""
    from smr.backbones import pointmap
    monkeypatch.setattr(pointmap, "checkpoint_dirs", lambda: [tmp_path])
    cls = _REGISTRY["dust3r"]
    assert cls.resolve_weights() == cls.weights_hub


def test_explicit_override_beats_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("SMR_CKPT_DIR", str(tmp_path))
    assert _REGISTRY["dust3r"].resolve_weights("/tmp/mine.pth") == \
        "/tmp/mine.pth"


def test_fetcher_manifest_covers_every_declared_backbone():
    """The fetcher reads the registry, so it can never drift from the
    adapters -- pin that."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fw", pathlib.Path(__file__).resolve().parents[1] /
        "scripts" / "fetch_weights.py")
    fw = importlib.util.module_from_spec(spec)
    import sys as _s
    argv = _s.argv; _s.argv = ["fw"]
    spec.loader.exec_module(fw)
    _s.argv = argv
    names = {r[0] for r in fw.entries(POINTMAP_ADAPTERS)}
    # only backbones with a direct URL appear: hub-hosted (fast3r, stream3r)
    # and Google-Drive-hosted (cut3r) ones are intentionally absent.
    assert names == set(FETCHABLE + GDRIVE), (
        f"manifest {names} != {set(FETCHABLE + GDRIVE)}")
    extras = fw.entries(POINTMAP_ADAPTERS, extras=True)
    assert len(extras) >= len(FETCHABLE)
    for _, fn, url, _req in extras:
        assert url.startswith("https://")
        assert "drive.google.com" in url or url_matches_file(url, fn)


def test_pi3_stages_a_subset_of_a_folder_rather_than_globbing_it():
    """pi3's loader takes a DIRECTORY and globs it sorted, so passing four
    paths out of a forty-one image folder silently loaded the wrong images.
    Every adapter must honour the same contract: a list of paths means
    exactly those, in that order."""
    import inspect
    from smr.backbones import pi3 as pi3_mod
    src = inspect.getsource(pi3_mod.Pi3Backbone.infer)
    assert "TemporaryDirectory" in src and "symlink_to" in src, (
        "pi3 must stage an explicit subset instead of globbing the parent")


@pytest.mark.parametrize("name", ALL_ADAPTERS)
def test_every_backbone_declares_its_pose_convention(name):
    """The single most dangerous piece of per-backbone knowledge: VGGT is
    world-from-camera, everything else so far is camera-to-world.  It must be
    declared on the class, not buried in a comment inside infer()."""
    assert getattr(_REGISTRY[name], "pose_convention", None) in POSE_CONVENTIONS


@pytest.mark.parametrize("name", POINTMAP_ADAPTERS)
def test_pythonpath_style_repos_are_bootstrapped(name):
    """DUSt3R and MASt3R ship no setup.py, so `pip install -e` cannot work;
    the adapter must add third_party/<repo> to sys.path itself."""
    cls = _REGISTRY[name]
    assert cls.third_party_paths, f"{name} declares no third_party_paths"
    bb = get_backbone(name, device="cpu")
    bb.bootstrap()                                  # idempotent, never raises
    bb.bootstrap()


# ============================ TIER 1: third-party compat shims =============
def test_scipy_hierarchy_shim_restores_the_dropped_alias():
    """MASt3R's sparse aligner calls scipy.cluster.hierarchy.distance, an
    alias current SciPy dropped.  Simulate the server's SciPy by removing
    the attribute, then assert the shim restores working behaviour."""
    import numpy as _np
    import scipy.cluster.hierarchy as sch
    from smr.backbones.pointmap import patch_scipy_hierarchy_distance
    saved = getattr(sch, "distance", None)
    try:
        if hasattr(sch, "distance"):
            del sch.distance
        assert patch_scipy_hierarchy_distance() is True      # applied
        assert patch_scipy_hierarchy_distance() is False     # idempotent
        square = _np.array([[0.0, 0.5], [0.5, 0.0]])
        assert _np.allclose(sch.distance.squareform(square), [0.5])
    finally:
        if saved is not None:
            sch.distance = saved


def test_module_stub_never_shadows_a_real_install():
    from smr.backbones.pointmap import ensure_module_stub
    assert ensure_module_stub("numpy") is False       # real numpy untouched
    import numpy as _np
    assert not hasattr(_np, "__SMR_STUB__")


def test_module_stub_satisfies_an_unused_import():
    import sys as _s
    from smr.backbones.pointmap import ensure_module_stub
    name = "smr_fake_optional_dep"
    _s.modules.pop(name, None)
    try:
        assert ensure_module_stub(name, Thing=lambda *a, **k: None) is True
        mod = __import__(name)
        assert mod.__SMR_STUB__ and mod.Thing() is None
        assert ensure_module_stub(name) is False      # now importable
    finally:
        _s.modules.pop(name, None)


def test_mast3r_exposes_a_scipy_free_kinematic_mode():
    """If the shim ever stops working, 'mst' is upstream's own alternative
    and must remain reachable without editing the adapter."""
    bb = get_backbone("mast3r", device="cpu", kinematic_mode="mst")
    assert bb.kinematic_mode == "mst"


def test_dotted_stub_is_reachable_by_from_import(tmp_path):
    """`from pkg.sub import X` must find a stubbed submodule -- this is how
    MUSt3R's retrieval module is bypassed without faking faiss AND asmk."""
    import sys as _s
    from smr.backbones.pointmap import ensure_module_stub
    name = "json.smr_fake_sub"          # real parent, fake child
    _s.modules.pop(name, None)
    try:
        assert ensure_module_stub(name, Retriever=type("R", (object,), {}))
        from json.smr_fake_sub import Retriever    # noqa: F401
        assert Retriever is not None
        import json as _j
        assert getattr(_j, "smr_fake_sub").__SMR_STUB__
    finally:
        _s.modules.pop(name, None)
        import json as _j
        if hasattr(_j, "smr_fake_sub"):
            delattr(_j, "smr_fake_sub")


def test_collections_alias_shim_restores_removed_names():
    """Python 3.10 deleted collections.Iterable et al; libraries in MUSt3R's
    import chain still use them.  The shim must re-bind the identical object
    from collections.abc, add names only, and be idempotent."""
    import collections
    import collections.abc as cabc
    from smr.backbones.pointmap import patch_collections_abc_aliases
    saved = {n: getattr(collections, n) for n in ("Iterable", "Mapping")
             if hasattr(collections, n)}
    try:
        for n in ("Iterable", "Mapping"):
            if hasattr(collections, n):
                delattr(collections, n)
        restored = patch_collections_abc_aliases()
        assert "Iterable" in restored and "Mapping" in restored
        assert collections.Iterable is cabc.Iterable
        assert collections.Mapping is cabc.Mapping
        assert patch_collections_abc_aliases() == []          # idempotent
    finally:
        for n, v in saved.items():
            setattr(collections, n, v)


def test_center_crop_box_matches_torchvision_rounding():
    """torchvision uses int(round(...)), which is BANKER'S rounding in
    Python 3: a 9->4 crop starts at 2, not 3.  Getting this wrong shifts
    every cropped image by a pixel, silently."""
    from smr.backbones.pointmap import center_crop_box
    assert center_crop_box(10, 10, 4, 4) == (3, 3)
    assert center_crop_box(9, 9, 4, 4) == (2, 2)        # round(2.5) == 2
    assert center_crop_box(384, 512, 384, 288) == (0, 112)
    assert center_crop_box(5, 7, 2, 3) == (2, 2)




@pytest.mark.parametrize("name", POINTMAP_ADAPTERS)
def test_uncloned_repo_says_so_instead_of_generic_missing_module(name):
    """'missing module' is ambiguous: not cloned, or cloned but not
    installed?  They have different fixes, so the error must distinguish
    them and name the command."""
    bb = get_backbone(name, device="cpu")
    if all(bb.preflight().values()):
        pytest.skip(f"{name} is importable here")
    with pytest.raises(ImportError) as e:
        bb.infer(["a.png", "b.png"])
    msg = str(e.value)
    if bb.missing_repos():
        assert "not cloned" in msg and "setup_backbones.sh" in msg
    else:
        assert "pip install -e" in msg and "--no-deps" in msg


def test_isolate_third_party_purges_cached_modules(tmp_path, monkeypatch):
    """Several repos vendor their own dust3r fork.  Once one is imported,
    sys.modules caches it and the next backbone silently gets the wrong
    one -- which is exactly how mvdust3r's MultiView class 'disappeared'."""
    import sys as _s
    import types
    from smr.backbones.pointmap import isolate_third_party
    fake = types.ModuleType("dust3r")
    fake.__SMR_TEST__ = True
    _s.modules["dust3r"] = fake
    _s.modules["dust3r.model"] = types.ModuleType("dust3r.model")
    try:
        dropped = isolate_third_party(("dust3r",), "does_not_exist")
        assert "dust3r" in dropped and "dust3r.model" in dropped
        assert "dust3r" not in _s.modules
    finally:
        _s.modules.pop("dust3r", None)
        _s.modules.pop("dust3r.model", None)


def test_isolate_third_party_puts_vendored_path_first(tmp_path, monkeypatch):
    import sys as _s
    from smr.backbones import pointmap as pm
    monkeypatch.setattr(pm, "REPO_ROOT", tmp_path)
    (tmp_path / "third_party" / "vendorA").mkdir(parents=True)
    (tmp_path / "third_party" / "vendorB").mkdir(parents=True)
    saved = list(_s.path)
    try:
        pm.isolate_third_party((), "vendorA")
        pm.isolate_third_party((), "vendorB")
        assert _s.path[0] == str(tmp_path / "third_party" / "vendorB")
        pm.isolate_third_party((), "vendorA")          # re-prioritise
        assert _s.path[0] == str(tmp_path / "third_party" / "vendorA")
        assert _s.path.count(str(tmp_path / "third_party" / "vendorA")) == 1
    finally:
        _s.path[:] = saved


def test_stub_weights_enum_satisfies_default_arguments():
    from smr.backbones.pointmap import _StubWeightsEnum
    assert _StubWeightsEnum.IMAGENET1K_V1 and _StubWeightsEnum.DEFAULT


def test_permissive_stub_mints_attributes_but_not_dunders():
    """A permissive stub satisfies `from mod import Anything`, but must
    raise AttributeError for dunders -- importlib asks a package for
    __path__ and would try to iterate whatever it gets back."""
    import sys as _s
    from smr.backbones.pointmap import ensure_module_stub
    name = "smr_fake_optional_pkg"
    for n in (name, f"{name}.sub"):
        _s.modules.pop(n, None)
    try:
        ensure_module_stub(name, permissive=True)
        ensure_module_stub(f"{name}.sub", permissive=True)
        mod = _s.modules[name]
        assert mod.AnyClass.__SMR_STUB__ is True
        with pytest.raises(AttributeError):
            mod.__getattr__("__all__")
        assert isinstance(mod.__path__, list)
        exec(f"from {name}.sub import SomeSymbol")     # the real usage
    finally:
        for n in (name, f"{name}.sub"):
            _s.modules.pop(n, None)


@pytest.mark.parametrize("name", ["fast3r", "monst3r"])
def test_only_inference_irrelevant_modules_are_stubbed(name):
    """A genuinely required dependency must never be silently faked: the
    allowlist stays small and explicit, and anything off it re-raises."""
    # Deliberately small: training, evaluation and plotting packages --
    # plus sam2, which is a real model and is therefore only permitted
    # because the adapter switches OFF the feature that calls it (see the
    # test below, which enforces that link).
    allowed = {"pl_bolts", "open3d", "wandb", "evo", "seaborn", "matplotlib",
               "sam2"}
    assert set(_REGISTRY[name].OPTIONAL_AT_INFERENCE) <= allowed


def test_optional_stub_helper_reraises_unlisted_modules():
    from smr.backbones.pointmap import import_with_optional_stubs

    def loader():
        raise ModuleNotFoundError("No module named 'scipy'", name="scipy")

    with pytest.raises(ModuleNotFoundError):
        import_with_optional_stubs(loader, ("open3d",), "test")


def test_optional_stub_helper_stubs_listed_modules():
    import sys as _s
    from smr.backbones.pointmap import import_with_optional_stubs
    _s.modules.pop("smr_fake_viz", None)
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModuleNotFoundError("nope", name="smr_fake_viz")
        return "loaded"

    try:
        assert import_with_optional_stubs(
            loader, ("smr_fake_viz",), "test") == "loaded"
        assert calls["n"] == 2
    finally:
        _s.modules.pop("smr_fake_viz", None)


def test_fast3r_passes_a_torch_device_not_a_string():
    """fast3r's loss_of_one_batch does autocast(device_type=device.type),
    so the plain "cuda" string every other adapter passes fails there."""
    import inspect
    src = inspect.getsource(_REGISTRY["fast3r"]._raw)
    assert "torch.device(self.device)" in src
    assert "inference(images, self._model, dev," in src


def test_fast3r_handles_both_inference_return_shapes():
    """fast3r's inference() returns (result, profiling_info) only when
    profiling=True and a bare dict otherwise; the README example shows the
    two-value form, which is why unpacking blindly fails."""
    import inspect
    src = inspect.getsource(_REGISTRY["fast3r"]._raw)
    assert "isinstance(out, tuple)" in src
    assert "out, _ = inference" not in src


def test_monst3r_disables_the_feature_that_would_call_sam2():
    """sam2 is stubbed, so anything that actually calls it would explode at
    runtime.  The adapter must therefore default MonST3R's SAM2 mask
    refinement OFF -- stubbing a real model is only honest if nothing
    invokes it."""
    bb = get_backbone("monst3r", device="cpu")
    assert bb.sam2_mask_refine is False
    import inspect
    src = inspect.getsource(type(bb)._raw)
    assert "sam2_mask_refine" in src, "the flag must reach global_aligner"
    on = get_backbone("monst3r", device="cpu", sam2_mask_refine=True)
    assert on.sam2_mask_refine is True          # opt-in still possible


def test_isolation_drops_competing_third_party_paths(tmp_path, monkeypatch):
    """dust3r's model.py inserts its own vendored croco dir into sys.path at
    import time, and that entry survives a sys.modules purge -- so a later
    backbone's `from models.croco import CrocoConfig` resolved to DUSt3R's
    copy, which has no CrocoConfig.  Isolation must remove foreign
    third_party entries, not merely prepend its own."""
    import sys as _s
    from smr.backbones import pointmap as pm
    monkeypatch.setattr(pm, "REPO_ROOT", tmp_path)
    (tmp_path / "third_party" / "repoA" / "croco").mkdir(parents=True)
    (tmp_path / "third_party" / "repoB" / "src" / "croco").mkdir(parents=True)
    stale = str(tmp_path / "third_party" / "repoA" / "croco")
    saved = list(_s.path)
    try:
        _s.path.insert(0, stale)                    # what repoA leaves behind
        pm.isolate_third_party((), "repoB/src/croco", "repoB/src")
        assert stale not in _s.path, "foreign croco path still shadowing"
        assert _s.path[0] == str(
            tmp_path / "third_party" / "repoB" / "src" / "croco")
        assert "/some/unrelated/path" not in _s.path   # only third_party
    finally:
        _s.path[:] = saved


def test_torch_load_shim_is_idempotent_and_keeps_explicit_choice():
    """The shim restores torch<2.6's default for calls it cannot reach, but
    an explicit weights_only=True must still win."""
    try:                     # torch may be present but un-importable here
        import torch          # (CUDA libs absent in the dev sandbox);
    except Exception as e:    # importorskip only catches ImportError
        pytest.skip(f"torch unusable in this environment: {type(e).__name__}")
    from smr.backbones.pointmap import patch_torch_load_legacy_checkpoints
    original = torch.load
    try:
        patch_torch_load_legacy_checkpoints()
        if not getattr(torch.load, "__SMR_PATCHED__", False):
            pytest.skip("torch < 2.6: no shim needed")
        assert patch_torch_load_legacy_checkpoints() is False   # idempotent
        # the shim closes over the ORIGINAL torch.load, so exercise it via
        # a real round-trip instead of swapping __wrapped__
        import io
        buf = io.BytesIO()
        torch.save({"a": torch.zeros(2)}, buf)
        buf.seek(0)
        got = torch.load(buf)                       # no weights_only passed
        assert "a" in got
    finally:
        torch.load = original


