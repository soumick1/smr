"""Stitcher tests on a rendered two-lap world with known ground truth.

The world is small (2 x 32 frames at 96 px) so the whole file runs in a
few seconds on CPU.  Descriptors come from the RENDERED images through the
same function used on real captures, so anchor proposal is exercised
through appearance rather than an oracle.
"""
import numpy as np
import pytest

from smr.eval.trajectory import ate_rmse, auc_split, loop_closure_error
from smr.stitch import (AnchoredStitcher, DescriptorIndex, PassCache,
                        ScaffoldIndex, SimulatedRunner, make_chunks,
                        posegraph, probe, regauge, stitch_chained)
from smr.stitch import sim3
from smr.stitch.simulate import LoopWorld


@pytest.fixture(scope="module")
def world():
    return LoopWorld(n_per_lap=32, laps=2, H=96, W=96, seed=0, jitter=0.02)


@pytest.fixture(scope="module")
def chunks(world):
    return make_chunks(len(world.gt), 16, 8)


def test_rendered_descriptors_prefer_revisits(world):
    """A lap-two view must be closer to its lap-one counterpart than to the
    opposite side of the room -- otherwise appearance proposal is void."""
    D = world.descriptors @ world.descriptors.T
    n = world.n_per_lap
    rev = np.median([D[i, i + n] for i in range(n)])
    opp = np.median([D[i, (i + n // 2) % n] for i in range(n)])
    assert rev > 0.9 and rev > opp + 0.5


def test_zero_sites_reduces_exactly_to_chaining(world, chunks):
    cache = PassCache()
    runner = world.runner(noise=0.01, distortion=0.1, seed=1)
    est_ch = stitch_chained(chunks, cache, runner)
    r = AnchoredStitcher(DescriptorIndex(), n_sites=0).run(
        chunks, cache, runner, world.descriptors)
    assert np.allclose(r["est"], est_ch, atol=1e-12)
    assert r["n_loops"] == 0
    # ... and no extra pass was run
    assert cache.n_runs == len(chunks)


def test_anchoring_bounds_drift_where_chaining_compounds(world, chunks):
    """The claim under test, on a world where lap two revisits lap one:
    loop-closure error and cross-pass AUC must improve over chaining, and
    the batch solve on the same edges must not be worse than chaining."""
    pairs = world.revisit_pairs()
    wins = 0
    for seed in (1, 2, 3):
        cache = PassCache()
        runner = world.runner(noise=0.01, distortion=0.1, seed=seed)
        est_ch = stitch_chained(chunks, cache, runner)
        r = AnchoredStitcher(ScaffoldIndex(seed=0), n_sites=2,
                             correction="distribute").run(
            chunks, cache, runner, world.descriptors)
        assert r["n_loops"] >= 2
        lc_ch = loop_closure_error(est_ch, world.gt, pairs)["rot_deg_mean"]
        lc_sm = loop_closure_error(r["est"], world.gt, pairs)["rot_deg_mean"]
        ax_ch = auc_split(est_ch, world.gt, chunks)["auc_cross"]
        ax_sm = auc_split(r["est"], world.gt, r["passes"])["auc_cross"]
        pg, info = posegraph.solve(r, chunks)
        assert info["n_loop_edges"] >= 1
        if lc_sm < lc_ch and ax_sm > ax_ch and \
                ate_rmse(pg, world.gt) <= ate_rmse(est_ch, world.gt):
            wins += 1
    assert wins >= 2, "anchoring must beat chaining on most seeds"


def test_false_site_is_rejected_by_geometry(world, chunks):
    """A proposed site whose stored relative pose disagrees with the pass
    must fail verification -- one wrong loop closure must not move a chunk."""
    st = AnchoredStitcher(DescriptorIndex(), n_sites=2)
    P = np.tile(np.eye(4), (4, 1, 1))
    P[1, :3, 3] = [1, 0, 0]
    P[3, :3, 3] = [0, 1, 0]
    P[3, :3, :3] = sim3.rotmat([0, 0, 0.5])
    stored = {10: P[2], 11: P[3]}
    pos = {10: 2, 11: 3}
    ok, rot, dr = st._verify_site((10, 11, 0.0, 0.0), P, pos, stored)
    assert ok and rot < 1e-9
    bad = {10: P[2], 11: np.linalg.inv(P[3])}      # a wrong stored pose
    ok2, rot2, _ = st._verify_site((10, 11, 0.0, 0.0), P, pos, bad)
    assert not ok2 and rot2 > 10


def test_plain_index_finds_the_same_loops(world, chunks):
    """Prediction recorded in advance: the scaffold address is not what
    buys the trajectory result; a plain descriptor index closes the same
    loops.  The address earns its keep elsewhere (rendering, imagination,
    novelty)."""
    cache = PassCache()
    runner = world.runner(noise=0.01, distortion=0.1, seed=2)
    r_s = AnchoredStitcher(ScaffoldIndex(seed=0), n_sites=2).run(
        chunks, cache, runner, world.descriptors)
    r_p = AnchoredStitcher(DescriptorIndex(), n_sites=2).run(
        chunks, cache, runner, world.descriptors)
    assert abs(r_s["n_loops"] - r_p["n_loops"]) <= 1


def test_template_address_matches_bump_dynamics():
    """The pilot encodes addresses analytically; the real bumps settle to
    the same address up to the decode floor."""
    from smr.dynamics import ScaffoldState
    from smr.stitch.memory_index import PERIODS
    from smr.utils.geometry import euler_zyx_to_R, make_T
    ss = ScaffoldState(periods=list(PERIODS), ring_N=128, torus_N=32, seed=0,
                       omega_max=0.16)
    ss.calibrate()
    tpl = ScaffoldIndex(torus_N=32, N_h=1024, seed=0, encode="template")
    dyn = ScaffoldIndex(torus_N=32, N_h=1024, seed=0, encode="dynamics",
                        scaffold_state=ss)
    rng = np.random.default_rng(0)
    T = make_T(euler_zyx_to_R(*rng.uniform(-0.5, 0.5, 3)), rng.uniform(-1.5, 1.5, 3))
    h_t, _ = tpl.address(T)
    h_d, _ = dyn.address(T)
    T2 = make_T(T[:3, :3], T[:3, 3] + np.array([0.6, 0.0, 0.0]))
    h_far, _ = tpl.address(T2)
    assert float(h_t @ h_d) > 0.9
    assert float(h_t @ h_d) > float(h_t @ h_far) + 0.3


def test_pass_cache_memoises_and_persists(tmp_path):
    gt = np.tile(np.eye(4), (6, 1, 1))
    gt[:, :3, 3] = np.arange(6)[:, None] * np.array([1.0, 0.2, 0.0])
    calls = []

    def runner(idx):
        calls.append(tuple(idx))
        return regauge(gt[idx]), 0.1, 0.0

    c = PassCache(tmp_path / "cache.npy")
    a = c.get([0, 1, 2], runner)
    b = c.get([0, 1, 2], runner)
    assert a is b and len(calls) == 1
    c2 = PassCache(tmp_path / "cache.npy")          # reload from disk
    assert c2.has([0, 1, 2]) and not c2.has([1, 2, 3])
    secs, peak = c2.totals([[0, 1, 2]])
    assert abs(secs - 0.1) < 1e-9


def test_simulated_runner_regauges_and_zero_error_is_exact():
    gt = np.tile(np.eye(4), (5, 1, 1))
    for i in range(5):
        gt[i, :3, :3] = sim3.rotmat([0, 0.3 * i, 0])
        gt[i, :3, 3] = [np.cos(0.3 * i), 0, np.sin(0.3 * i)]
    p, _, _ = SimulatedRunner(gt, noise=0.0, distortion=0.0)([0, 1, 2, 3, 4])
    assert np.allclose(p[0], np.eye(4))
    assert abs(np.median(np.linalg.norm(p[1:, :3, 3], axis=1)) - 1.0) < 1e-12


def test_gate_compares_chunks_with_reference():
    rows = [dict(auc30=70.0), dict(auc30=65.0), dict(auc30=80.0)]
    ok, d = probe.gate(rows, dict(auc30=75.0), ratio=0.6, floor=40.0)
    assert ok and d["median_auc"] == 70.0
    ok2, _ = probe.gate([dict(auc30=20.0), dict(auc30=90.0), dict(auc30=90.0)],
                        dict(auc30=75.0), ratio=0.6, floor=40.0)
    assert not ok2
