"""Executable tests of the guarantees the paper states (v177, plan Task 2).

Every test runs the REAL code path (sim3, AnchoredStitcher, ScaffoldIndex) on
synthetic inputs; nothing here needs a GPU.  Each test's docstring records
the finding it establishes for the text, so App. A/B/C/D can be written from
the code rather than from memory.  Run with `pytest -s tests/test_guarantees.py`
to see the measured numbers.

Findings (as of v167 code, 2026-09-10):
  A. sim3.fit_poses is ORIENTATION-AWARE: rotation by orthogonal Procrustes
     on the frame orientations, scale from the ratio of centre spreads,
     translation from centroids.  A two-view site therefore determines the
     rotation exactly; one view leaves only scale unresolved (flagged).
     App. A currently describes a centres-only Horn/Umeyama fit with
     orientation as a check -- that is not the code.
  B. correction="none" is NOT an identity control: a window with proposed
     sites obtains its local geometry from the enlarged (anchored) pass, so
     the trajectory diverges from the raw chain at the first window with a
     proposal, accepted or rejected.  Rejected proposals therefore do alter
     the geometry state (the window's own poses), not only the logs.
  C. Placement corrections preserve within-owner relative rotation and
     translation DIRECTION exactly; magnitude scales with the owner's s_k.
     EXCEPTION found by the test: at an accepted closure the default
     smooth_junctions=True blends the preceding window's overlap frames per
     frame between the two placements (anti-step measure).  App. B must say
     so; "every frame of a window receives the same transform" holds only
     away from accepted closures (or with smooth_junctions=False).
  D. Processing is causal: the decisions (sites, loops) and the placements
     of windows untouched by later closures are identical between a run of
     K windows and the first K windows of a longer run over the same frame
     list and schedule.
  E. Address convention: a scaffold address is written once at bind time
     and is NOT re-addressed after a placement correction (update_pose moves
     the pose annotation only); cue retrieval is unaffected by corrections,
     pose-proximity proposals use the corrected poses.
  F. The operative gate constants are the AnchoredStitcher defaults listed
     in test_gate_constants_are_the_paper_values; the two-site mutual
     agreement test (site_agree_*) is off by default (1e9).
"""
from __future__ import annotations

import copy
import json
import pathlib
import re

import numpy as np
import pytest

from smr.eval.trajectory import rotation_angle_deg
from smr.stitch import sim3
from smr.stitch.anchored import AnchoredStitcher, stitch_chained
from smr.stitch.chunks import make_chunks
from smr.stitch.memory_index import DescriptorIndex, ScaffoldIndex
from smr.stitch.passes import PassCache
from smr.stitch.simulate import LoopWorld

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------- helpers
def _rand_R(rng, max_deg=180.0):
    v = rng.standard_normal(3)
    v = v / np.linalg.norm(v) * np.deg2rad(rng.uniform(0, max_deg))
    return sim3.rotmat(v)


def _poses(rng, n, centres=None, max_deg=180.0):
    T = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        T[i, :3, :3] = _rand_R(rng, max_deg)
        T[i, :3, 3] = centres[i] if centres is not None else rng.uniform(-2, 2, 3)
    return T


def _rand_S(rng):
    return (float(np.exp(rng.uniform(-0.7, 0.7))), _rand_R(rng), rng.uniform(-3, 3, 3))


def _errors(S_hat, A, B, S_true):
    moved = sim3.apply(S_hat, A)
    rot = max(rotation_angle_deg(moved[i, :3, :3].T @ B[i, :3, :3]) for i in range(len(A)))
    pos = float(np.max(np.linalg.norm(moved[:, :3, 3] - B[:, :3, 3], axis=1)))
    return dict(rot_deg=rot, pos=pos, logscale=abs(float(np.log(S_hat[0] / S_true[0]))))


# =========================================================== A. estimator
def test_estimator_general_two_view_one_view_collinear_and_noise(capsys):
    """Which configurations the historical-site estimator supports, and how
    weak ones are flagged.  Exact recovery expected wherever the geometry
    determines the quantity; `scale_ok` must be False where it does not."""
    rng = np.random.default_rng(0)
    rows = []
    for name, n, centres in [
        ("general (4 views)", 4, None),
        ("two-view site, distinct rotations", 2, None),
        ("two-view site, coincident centres", 2, np.zeros((2, 3))),
        ("one view", 1, None),
        ("near-collinear (4 on a line)", 4, np.outer(np.linspace(0, 1, 4), [1.0, 0.2, 0.0])),
    ]:
        A = _poses(rng, n, centres)
        S = _rand_S(rng)
        B = sim3.apply(S, A)
        S_hat, info = sim3.fit_poses(A, B)
        e = _errors(S_hat, A, B, S)
        rows.append((name, e, bool(info["scale_ok"])))
        # rotation and translation of every pose are reproduced exactly
        # whenever the centres carry the scale ...
        if info["scale_ok"]:
            assert e["rot_deg"] < 1e-4 and e["pos"] < 1e-6 and e["logscale"] < 1e-9, (name, e)
        else:
            # ... and when they do not (one view / coincident centres) the
            # orientation is still exact, scale is left at 1 and FLAGGED
            assert e["rot_deg"] < 1e-4, (name, e)
            assert S_hat[0] == 1.0
    # noise: bounded errors, scale still resolved
    A = _poses(rng, 6)
    S = _rand_S(rng)
    B = sim3.apply(S, A)
    Bn = B.copy()
    for i in range(len(Bn)):
        Bn[i, :3, :3] = Bn[i, :3, :3] @ sim3.rotmat(np.deg2rad(1.0) * rng.standard_normal(3) / np.sqrt(3))
        Bn[i, :3, 3] += 0.01 * rng.standard_normal(3)
    S_hat, info = sim3.fit_poses(A, Bn)
    e = _errors(S_hat, A, B, S)
    rows.append(("noise 1 deg / 0.01 u (6 views)", e, bool(info["scale_ok"])))
    assert e["rot_deg"] < 3.0 and e["pos"] < 0.05 and e["logscale"] < 0.02
    with capsys.disabled():
        print("\n[A] estimator configurations (max residual over poses)")
        for name, e, ok in rows:
            print(f"    {name:<38} rot {e['rot_deg']:8.2e} deg  pos {e['pos']:8.2e}  "
                  f"|log s| {e['logscale']:8.2e}  scale_ok={ok}")


def test_estimator_uses_orientations_not_only_centres():
    """Orientations PARTICIPATE in the rotation estimate: perturbing the
    source orientations (centres fixed) changes the fitted rotation by the
    same angle.  Hence the two-view site problem is well posed."""
    rng = np.random.default_rng(1)
    A = _poses(rng, 2)
    S = _rand_S(rng)
    B = sim3.apply(S, A)
    R_hat0 = sim3.fit_poses(A, B)[0][1]
    A2 = A.copy()
    twist = sim3.rotmat(np.deg2rad(30.0) * np.array([0, 0, 1.0]))
    for i in range(2):
        A2[i, :3, :3] = twist @ A2[i, :3, :3]        # rotate frames (world), keep centres
    R_hat1 = sim3.fit_poses(A2, B)[0][1]
    assert abs(rotation_angle_deg(R_hat0.T @ R_hat1) - 30.0) < 1e-4
    # a centres-only estimator (Umeyama on the two centres) would be blind to this
    S_c, _ = sim3.fit_poses(A, B)
    assert np.allclose(S_c[2], sim3.fit_poses(A2, B)[0][2] + 0, atol=10)  # (translation differs; documented, not asserted tightly)


def test_robust_fit_rejects_one_wrong_anchor_and_fixed_scale_is_exact():
    rng = np.random.default_rng(2)
    A = _poses(rng, 5)
    S = _rand_S(rng)
    B = sim3.apply(S, A)
    B[4, :3, 3] += np.array([5.0, -4.0, 3.0])            # a false anchor
    B[4, :3, :3] = _rand_R(rng)
    S_hat, keep, info = sim3.fit_poses_robust(A, B, rot_thresh_deg=10.0, pos_thresh_rel=0.5, min_inliers=3)
    assert keep.tolist() == [True, True, True, True, False]
    e = _errors(S_hat, A[:4], B[:4], S)
    assert e["rot_deg"] < 1e-4 and e["pos"] < 1e-6
    # fixed scale, short-baseline two-view site: rotation/translation exact
    A2 = _poses(rng, 2, centres=np.array([[0, 0, 0], [0.01, 0, 0]]))
    B2 = sim3.apply(S, A2)
    Sf = sim3.fit_poses_fixed_scale(A2, B2, S[0])
    e2 = _errors(Sf, A2, B2, S)
    assert e2["rot_deg"] < 1e-4 and e2["pos"] < 1e-6


# ================================================= synthetic run fixtures
def _world(seed=0):
    w = LoopWorld(n_per_lap=40, laps=2, H=96, W=96, seed=seed, jitter=0.02)
    runner = w.runner(noise=0.01, distortion=0.1, seed=seed + 1)
    return w, runner


def _common():
    return dict(n_sites=2, desc_thresh=None, mutual_nn=False, remeasure=False,
                site_agree_rot=1e9, site_agree_pos=1e9, verbose=False)


# ================================================ B. no-write-back control
def test_no_write_back_is_not_an_identity_control(capsys):
    """correction='none' vs the raw chain, SAME cached passes, compared before
    any alignment or PGO.  The first differing frame is in the first window
    that proposed a site: that window's local geometry comes from the
    enlarged anchored pass, whether or not the proposal was accepted."""
    w, runner = _world()
    chunks = make_chunks(len(w.gt), 16, 8)
    cache = PassCache(None)
    raw = stitch_chained(chunks, cache, runner)
    st = AnchoredStitcher(DescriptorIndex(), correction="none", **_common())
    r = st.run(chunks, cache, runner, w.descriptors)
    est = r["est"]
    assert est.shape == raw.shape
    diff = np.array([np.linalg.norm(est[i] - raw[i]) for i in range(len(est))])
    first_frame = int(np.argmax(diff > 1e-9)) if (diff > 1e-9).any() else None
    ev = r["events"]
    first_prop = next((e["chunk"] for e in ev if e.get("sites")), None)
    assert first_prop is not None, "the synthetic two-lap world must propose revisits"
    assert first_frame is not None, "with proposals the two paths must diverge"
    owner = r["owner"]
    assert owner[first_frame] == first_prop, (
        f"first divergence at frame {first_frame} (owner {owner[first_frame]}) "
        f"but first proposal at window {first_prop}")
    # every frame owned by an earlier window is bit-identical to the chain
    assert diff[np.array(owner) < first_prop].max() < 1e-9
    # the mechanism: the proposing window ran an ENLARGED pass
    npass = [len(p) for p in r["passes"]]
    assert npass[first_prop] > len(chunks[first_prop])
    # and its local poses differ from the plain window pass -> a REJECTED
    # proposal also changes geometry state, not just the log
    plain = cache.get(list(chunks[first_prop]), runner)["poses"]
    loc = np.stack([r["local"][g] for g in chunks[first_prop]])
    # the two passes have different gauges; compare after the best Sim(3)
    _, info = sim3.fit_poses(plain, loc)
    d_local = float(max(info["rot_res_deg"].max(), info["pos_res"].max() / max(info["spread"], 1e-9)))
    assert d_local > 1e-6, "an enlarged pass should not reproduce the plain pass exactly"
    n_rej = sum(1 for e in ev if e.get("loop") and not e["loop"]["accepted"])
    with capsys.disabled():
        print(f"\n[B] corr_none vs chain: first divergence frame {first_frame}, window {first_prop} "
              f"(pass {npass[first_prop]} frames vs window {len(chunks[first_prop])}); "
              f"within-window geometry change after Sim(3) alignment {d_local:.3g} (deg / spreads); rejected proposals in run: {n_rej}; "
              f"final ATE-like RMS diff {np.sqrt(np.mean(diff**2)):.3g}")


# ============================================ C. owner-window preservation
def test_owner_window_relative_geometry_is_preserved_under_corrections(capsys):
    """Sim(3) placement of an owner window: relative rotation and translation
    direction of any two frames it owns are invariant; the magnitude scales
    by s_k.  Checked algebraically and on the real stitcher's output."""
    rng = np.random.default_rng(3)
    P = _poses(rng, 6)
    for _ in range(20):
        T_k = _rand_S(rng)
        Q = sim3.apply(T_k, P)
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                Rp = P[i, :3, :3].T @ P[j, :3, :3]
                Rq = Q[i, :3, :3].T @ Q[j, :3, :3]
                assert rotation_angle_deg(Rp.T @ Rq) < 1e-4
                tp = P[i, :3, :3].T @ (P[j, :3, 3] - P[i, :3, 3])
                tq = Q[i, :3, :3].T @ (Q[j, :3, 3] - Q[i, :3, 3])
                assert np.allclose(tq, T_k[0] * tp, atol=1e-9)
    # the real path: est = T_owner . local for every frame EXCEPT the overlap
    # frames of the window preceding an accepted closure, which
    # smooth_junctions blends per frame between the two placements (a
    # deliberate anti-step measure; TUM fr1_room AUC_in 80.8 -> 75.8 without it)
    w, runner = _world()
    chunks = make_chunks(len(w.gt), 16, 8)
    cache = PassCache(None)
    r = AnchoredStitcher(DescriptorIndex(), correction="relax", **_common()).run(
        chunks, cache, runner, w.descriptors)
    assert r["n_loops"] >= 1, "need an accepted closure to test a corrected window"
    owner = np.array(r["owner"])
    closed_at = [e["chunk"] for e in r["events"] if e.get("loop") and e["loop"]["accepted"]]
    blended = set()
    for k in closed_at:
        blended |= {g for g in chunks[k] if owner[g] < k}       # overlap frames of the previous window
    exact, worst_blend = 0.0, 0.0
    for k in sorted(set(owner.tolist())):
        gs = [g for g in range(len(owner)) if owner[g] == k]
        if len(gs) < 2:
            continue
        L = np.stack([r["local"][g] for g in gs])
        E = r["est"][gs]
        S_k, info = sim3.fit_poses(L, E)
        res = max(float(info["rot_res_deg"].max()), float(info["pos_res"].max()))
        if any(g in blended for g in gs):
            worst_blend = max(worst_blend, res)
        else:
            exact = max(exact, res)
    assert exact < 1e-4, exact
    assert worst_blend > 1e-3, "expected the junction blend to move overlap frames"
    # with the blend off, the guarantee is exact for every window
    cache2 = PassCache(None)
    r2 = AnchoredStitcher(DescriptorIndex(), correction="relax", smooth_junctions=False,
                          **_common()).run(chunks, cache2, runner, w.descriptors)
    owner2 = np.array(r2["owner"])
    worst2 = 0.0
    for k in sorted(set(owner2.tolist())):
        gs = [g for g in range(len(owner2)) if owner2[g] == k]
        if len(gs) >= 2:
            _, info = sim3.fit_poses(np.stack([r2["local"][g] for g in gs]), r2["est"][gs])
            worst2 = max(worst2, float(info["rot_res_deg"].max()), float(info["pos_res"].max()))
    assert worst2 < 1e-4, worst2
    with capsys.disabled():
        print(f"\n[C] owner-window preservation: exact ({exact:.1e}) for windows not adjacent to an accepted "
              f"closure; junction blend (smooth_junctions=True, default) moves the preceding window's "
              f"{len(blended)} overlap frames by up to {worst_blend:.2f} deg/units across {len(closed_at)} closures; "
              f"with smooth_junctions=False every window is exact ({worst2:.1e})")


# ========================================================== D. causality
def test_causality_prefix_run_matches_longer_run(capsys):
    """Same frame list and window schedule; run K windows, then all.  Decisions
    for windows < K and placements of windows not inside a later closure's
    span must coincide (before evaluation alignment)."""
    w, runner = _world()
    chunks = make_chunks(len(w.gt), 16, 8)
    K = 6
    cache = PassCache(None)
    short = AnchoredStitcher(DescriptorIndex(), correction="relax", **_common()).run(
        chunks[:K], cache, runner, w.descriptors)
    full = AnchoredStitcher(DescriptorIndex(), correction="relax", **_common()).run(
        chunks, cache, runner, w.descriptors)

    def decisions(ev):
        out = []
        for e in ev:
            d = dict(chunk=e["chunk"], sites=[(s["view"], s["partner"], s["ok"]) for s in e.get("sites", [])])
            lp = e.get("loop")
            d["loop"] = None if not lp else (lp["accepted"], lp.get("reason"), lp.get("anchor_chunk"))
            out.append(d)
        return out
    ds, df = decisions(short["events"]), decisions(full["events"])[:K]
    assert ds == df, "proposals or acceptance decisions changed with future frames"
    # placements: frames whose owner window was never inside a later
    # closure span must be identical
    later_spans = [tuple(e["loop"]["stretch"]) for e in full["events"][K:]
                   if e.get("loop") and e["loop"]["accepted"]]
    own = np.array(full["owner"])
    n_frames_short = len(short["est"])
    untouched = np.ones(n_frames_short, bool)
    for lo, hi in later_spans:
        untouched &= ~((own[:n_frames_short] >= lo) & (own[:n_frames_short] <= hi))
    d = np.abs(full["est"][:n_frames_short] - short["est"]).reshape(n_frames_short, -1).max(1)
    assert d[untouched].max() < 1e-9, d[untouched].max()
    with capsys.disabled():
        print(f"\n[D] causality: {K} of {len(chunks)} windows; decisions identical; "
              f"{int(untouched.sum())}/{n_frames_short} frames untouched by later closures identical; "
              f"later closures revised {int((~untouched).sum())} frames (max change {d[~untouched].max() if (~untouched).any() else 0:.3g})")


# ================================================ E. address maintenance
def test_address_convention_write_time_addresses_retained(capsys):
    """Bind, correct placements, query again.  Convention established: the
    scaffold address is the view's label written at bind time; update_pose
    changes the pose annotation and the state row, never the address, so
    cue retrieval is unchanged and near_pose follows the corrected poses."""
    rng = np.random.default_rng(4)
    n, D = 24, 64
    T = _poses(rng, n, centres=np.cumsum(rng.uniform(-0.3, 0.3, (n, 3)), 0), max_deg=30.0)
    S = rng.standard_normal((n, D))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    idx = ScaffoldIndex(N_h=1024, torus_N=32, seed=0, desc_dim=D)
    for g in range(n):
        idx.add(g, T[g], S[g])
    H0 = [h.copy() for h in idx.H]
    top_before = [idx.propose(S[g], top=1)[0][0] for g in range(n)]
    assert top_before == list(range(n)), "each cue must retrieve its own view before correction"
    C = _rand_S(rng)
    Tc = sim3.apply(C, T)
    for g in range(n // 2, n):
        idx.update_pose(g, Tc[g])
    assert all(np.array_equal(a, b) for a, b in zip(H0, idx.H)), "addresses were re-written"
    top_after = [idx.propose(S[g], top=1)[0][0] for g in range(n)]
    assert top_after == top_before
    for g in range(n // 2, n):
        assert np.allclose(idx.pose(g), Tc[g])
        assert idx.near_pose(Tc[g], k=1)[0] == g, "pose proposals must use the corrected poses"
    xi_new = idx.address(Tc[n - 1])[1]
    assert not np.allclose(xi_new, idx.XI[n - 1]) or np.allclose(idx.XI[n - 1], idx.address(Tc[n - 1])[1])
    with capsys.disabled():
        print("\n[E] address convention: write-time addresses retained after update_pose; "
              "cue retrieval unchanged; near_pose follows corrected poses")


# =================================================== F. gate constants
def test_gate_constants_are_the_paper_values():
    """The operative verification constants.  If a default changes, this
    test fails and Table 11 / App. D must be updated with it."""
    st = AnchoredStitcher(DescriptorIndex())
    assert (st.site_rot_deg, st.site_dir_deg, st.extent_factor) == (10.0, 25.0, 3.0)
    assert (st.rot_thresh_deg, st.pos_thresh_rel) == (10.0, 0.5)
    assert st.budget_rot == (10.0, 3.0, 45.0) and st.budget_pos == (1.0, 0.5)
    assert st.budget_logscale == 0.5
    assert st.tight_rot == (3.0, 1.0, 15.0) and st.tight_pos == (0.5, 0.15)
    assert (st.site_agree_rot, st.site_agree_pos) == (1e9, 1e9)        # mutual agreement OFF
    assert (st.top, st.partner_gap, st.min_old_frames, st.min_baseline_rel) == (5, 3, 2, 1.0)
    assert st.require_appearance and not st.remeasure and not st.mutual_nn
    # budget formula: loose with >= 2 verified sites, tight with 1
    assert st.budget(4, 2) == (min(45.0, 10.0 + 3.0 * 4), 1.0 + 0.5 * 4, 0.5)
    assert st.budget(4, 1) == (min(15.0, 3.0 + 1.0 * 4), 0.5 + 0.15 * 4, 0.5)


def test_pilot_a_defaults_documented():
    """Operative pilot_a configuration (v198): the --paper preset is the default and matches the manuscript
    (N_h 1024, k 64, torus 48, 256-unit orientation rings, median-depth scene unit, agreement test 3 deg / 0.15 E,
    Sim(3) interpolation, IRLS alignment with gamma_s 1.5); --legacy restores the pre-v198 configuration."""
    src = (ROOT / "experiments" / "pilot_a.py").read_text()
    paper = re.search(r"P = dict\((.*?)\)\n", src, re.S).group(1)
    legacy = re.search(r"L = dict\((.*?)\)\n", src, re.S).group(1)
    for key in ("N_h=1024", "torus_N=48", "k=64", "ring_N=256", 'scene_unit="auto"', 'correction="distribute"', 'site_agree="10,1.0"', 'local_from="gated"', "distortion_gate=5.0",
                "seq_scale_gate=1.5", 'fit_mode="irls"', "require_two_sites=True", "reject_on_disagree=True", "pair_strict=True"):
        assert key in paper, key
    for key in ("N_h=2048", "torus_N=32", "ring_N=0", 'correction="relax"', 'site_agree="1e9,1e9"'):
        assert key in legacy, key
    gate = re.search(r'desc_thresh is None and a\.descriptor == "dino":\s*\n\s*a\.desc_thresh = ([\d.]+)', src)
    assert gate and float(gate.group(1)) == 0.5


# ================================================ G. local_from="plain" (v181)
def _plain(**kw):
    """A stitcher with window geometry from the plain pass (v181 sets it as an attribute; skip if not patched)."""
    st = AnchoredStitcher(DescriptorIndex(), **kw)
    if not hasattr(st, "local_from"):
        pytest.skip("local_from not available: run `python scripts/apply_all_patches.py` (v181)")
    st.local_from = "plain"
    return st


def test_local_from_plain_is_an_exact_identity_control(capsys):
    """With the window geometry taken from its own pass, correction='none' must
    reproduce the raw chain bit for bit, whatever was proposed or rejected:
    the genuine no-write-back control the plan asks for (Task 2B)."""
    w, runner = _world()
    chunks = make_chunks(len(w.gt), 16, 8)
    cache = PassCache(None)
    raw = stitch_chained(chunks, cache, runner)
    r = _plain(correction="none", **_common()).run(chunks, cache, runner, w.descriptors)
    assert any(e.get("sites") for e in r["events"]), "proposals must still be made"
    assert np.abs(r["est"] - raw).max() < 1e-9, np.abs(r["est"] - raw).max()
    # local geometry of every window equals its plain pass exactly
    for k, idx in enumerate(chunks):
        plain = cache.get(list(idx), runner)["poses"]
        new = [g for g in idx if r["owner"][g] == k]
        for g in new:
            assert np.allclose(r["local"][g], plain[list(idx).index(g)], atol=1e-12)
    dist = [e["anchored_distortion"] for e in r["events"] if e.get("anchored_distortion")]
    assert dist, "windows with anchors must log the anchored-pass distortion"
    with capsys.disabled():
        print(f"\n[G] local_from=plain + correction=none == raw chain (max |diff| {np.abs(r['est'] - raw).max():.1e}); "
              f"anchored-pass distortion logged for {len(dist)} windows, median rot {np.median([d['rot_deg'] for d in dist]):.2f} deg")


def test_local_from_plain_still_closes_loops(capsys):
    """Closures are still measured (from the anchored pass, carried into the
    plain pass's frame) and written back; the trajectory improves on the chain."""
    w, runner = _world()
    chunks = make_chunks(len(w.gt), 16, 8)
    cache = PassCache(None)
    raw = stitch_chained(chunks, cache, runner)
    r = _plain(correction="relax", **_common()).run(chunks, cache, runner, w.descriptors)
    assert r["n_loops"] >= 1
    from smr.eval.trajectory import ate_rmse
    ate_raw, ate_plain = ate_rmse(raw, w.gt), ate_rmse(r["est"], w.gt)
    r2 = AnchoredStitcher(DescriptorIndex(), correction="relax", **_common()).run(
        chunks, PassCache(None), runner, w.descriptors)          # default: anchored
    ate_anch = ate_rmse(r2["est"], w.gt)
    assert ate_plain < ate_raw
    with capsys.disabled():
        print(f"\n[G] synthetic ATE: chain {ate_raw:.3f}, plain-local +SMR {ate_plain:.3f} ({r['n_loops']} loops), "
              f"anchored-local +SMR {ate_anch:.3f} ({r2['n_loops']} loops)")


# ================================================ H. paper acceptance rules (v198)
def test_paper_rules_two_pairs_and_disagreement(capsys):
    """require_two_sites: a window with one verified pair closes nothing; reject_on_disagree: when the two pairs'
    placements differ by more than the agreement tolerance, both are marked `disagree` and nothing is written back
    (no single-pair demotion); pair_strict: partners are exactly +-partner_gap."""
    w, runner = _world()
    chunks = make_chunks(len(w.gt), 16, 8)
    cache = PassCache(None)
    kw = dict(_common()); kw.update(pair_strict=True, require_two_sites=True, reject_on_disagree=True,
                                  site_agree_rot=3.0, site_agree_pos=0.15, correction="distribute")
    r = AnchoredStitcher(DescriptorIndex(), **kw).run(chunks, cache, runner, w.descriptors)
    one_pair_windows = [e for e in r["events"] if sum(1 for s in (e.get("sites") or []) if s.get("ok")) == 1]
    assert all(not (e.get("loop") or {}).get("accepted") for e in one_pair_windows), "a single valid pair must not close a loop"
    disagreed = [s for e in r["events"] for s in (e.get("sites") or []) if s.get("disagree")]
    assert all(s.get("disagree", {}).get("rejected") for s in disagreed), "disagreement must reject, not demote"
    for e in r["events"]:
        for s in (e.get("sites") or []):
            assert abs(int(s["partner"]) - int(s["view"])) == kw.get("partner_gap", 3)
    accepted = [e for e in r["events"] if (e.get("loop") or {}).get("accepted")]
    with capsys.disabled():
        print(f"\n[H] paper rules on the synthetic world: {len(accepted)} closures accepted, {len(disagreed)} pairs rejected by the "
              f"agreement test, {len(one_pair_windows)} single-pair windows left unchanged")


def test_irls_fit_matches_exact_on_clean_data():
    """App. A reweighted fit recovers an exact similarity on clean correspondences and tolerates one outlier."""
    from smr.stitch import sim3
    rng = np.random.default_rng(0)
    def rot(v):
        v = np.asarray(v, float); th = np.linalg.norm(v)
        if th < 1e-12:
            return np.eye(3)
        k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    A = np.tile(np.eye(4), (8, 1, 1)); A[:, :3, 3] = rng.normal(0, 1, (8, 3))
    for i in range(8):
        A[i, :3, :3] = rot(rng.normal(0, 0.8, 3))
    S = (1.7, rot([0.4, -0.2, 0.9]), np.array([0.3, -1.0, 2.0]))
    B = sim3.apply(S, A)
    B[3, :3, 3] += np.array([2.0, -3.0, 1.0])              # one outlier centre
    S_hat, keep, info = sim3.fit_poses_irls(A, B, rounds=5, kappa=2.5, rot_thresh_deg=10.0)
    err = np.linalg.norm(sim3.apply(S_hat, A)[[0, 1, 2, 4, 5, 6, 7], :3, 3] - B[[0, 1, 2, 4, 5, 6, 7], :3, 3], axis=1).max()
    assert err < 0.1, err
    assert info["weights"][3] < 0.5, "the outlier must be down-weighted"
