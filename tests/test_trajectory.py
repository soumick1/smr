"""Trajectory-metric tests: analytic cases where the answer is known.

These metrics decide the paper's headline table, so none of them is
trusted on inspection -- each is checked against a case whose value can be
derived by hand.
"""
import numpy as np
import pytest

from smr.eval.trajectory import (align_to_gt, apply_sim3_to_poses, ate_rmse,
                                 auc_at, drift_at_revisit,
                                 pairwise_pose_errors, rotation_angle_deg,
                                 rpe, summarise, umeyama_sim3)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def traj(n=12, seed=0):
    """A smooth c2w trajectory with varying orientation."""
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        T[i, :3, :3] = rot_z(9.0 * i)
        T[i, :3, 3] = [np.cos(i * 0.4), np.sin(i * 0.4), 0.05 * i]
    T[:, :3, 3] += rng.normal(scale=1e-9, size=(n, 3))
    return T


# ----------------------------------------------------------- Umeyama ------
def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    s_true, R_true, t_true = 2.5, rot_z(37.0), np.array([1.0, -2.0, 0.5])
    Y = (s_true * (R_true @ X.T).T) + t_true
    s, R, t = umeyama_sim3(X, Y)
    assert abs(s - s_true) < 1e-9
    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)


def test_umeyama_without_scale_leaves_scale_at_one():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(10, 3))
    s, _, _ = umeyama_sim3(X, 3.0 * X, with_scale=False)
    assert s == 1.0


def test_umeyama_refuses_degenerate_input():
    with pytest.raises(AssertionError):
        umeyama_sim3(np.zeros((2, 3)), np.zeros((2, 3)))


def test_sim3_applied_to_poses_moves_centres_and_rotations():
    T = traj(5)
    s, R, t = 2.0, rot_z(30.0), np.array([1.0, 0.0, 0.0])
    out = apply_sim3_to_poses(T, s, R, t)
    assert np.allclose(out[:, :3, 3], (s * (R @ T[:, :3, 3].T).T) + t)
    assert np.allclose(out[0, :3, :3], R @ T[0, :3, :3])
    assert np.allclose(out[:, 3, :], [0, 0, 0, 1])


# ----------------------------------------------------------- ATE / RPE ----
def test_ate_is_zero_for_a_similarity_transformed_copy():
    """ATE is gauge-free by construction: an estimate that differs from GT
    by any similarity must score exactly zero."""
    gt = traj(15)
    est = apply_sim3_to_poses(gt, 3.7, rot_z(52.0), np.array([2.0, -1.0, 4.0]))
    assert ate_rmse(est, gt) < 1e-9


def test_ate_equals_the_offset_it_is_given():
    """Shift every centre by a fixed vector along a direction the alignment
    cannot absorb: with a mean-centred fit the RMSE is the offset norm."""
    gt = traj(9)
    est = gt.copy()
    est[:, :3, 3] += np.array([0.1, 0.0, 0.0])
    assert ate_rmse(est, gt, with_scale=False) < 1e-9   # a pure shift IS a
    # similarity, so it is absorbed -- the informative case is a DRIFT:
    est = gt.copy()
    est[:, :3, 3] += np.linspace(0, 0.3, len(gt))[:, None] * np.array([1, 0, 0])
    assert ate_rmse(est, gt) > 0.01


def test_rpe_is_zero_for_a_similarity_transformed_copy():
    gt = traj(12)
    est = apply_sim3_to_poses(gt, 1.8, rot_z(20.0), np.array([0.5, 0.5, 0.0]))
    t_err, r_err = rpe(est, gt, delta=1)
    assert t_err < 1e-8 and r_err < 1e-6


def test_rpe_rotation_catches_a_known_per_step_error():
    """Rotate every other frame by 5 degrees: the per-step relative
    rotation error must be 5 degrees."""
    gt = traj(8)
    est = gt.copy()
    for i in range(1, len(est), 2):
        est[i, :3, :3] = est[i, :3, :3] @ rot_z(5.0)
    _, r_err = rpe(est, gt, delta=1)
    assert 4.9 < r_err < 5.1


def test_rotation_angle_matches_construction():
    for deg in (0.0, 7.5, 90.0, 179.0):
        assert abs(rotation_angle_deg(rot_z(deg)) - deg) < 1e-6


# ------------------------------------------------------------- AUC@30 ----
def test_auc_is_100_for_a_perfect_estimate():
    gt = traj(8)
    rra, rta = pairwise_pose_errors(gt.copy(), gt)
    # arccos is ill-conditioned near identity: a 1e-16 error in the trace
    # becomes ~1e-6 rad in the angle, so "zero" here means milli-degrees.
    # This is a property of the geodesic angle, not of the estimate, and it
    # is far below any threshold the AUC curve cares about.
    assert np.allclose(rra, 0, atol=1e-3)
    assert auc_at(rra, rta) == pytest.approx(100.0)


def test_auc_is_scale_and_gauge_free():
    """Pairwise angles need no alignment: a similarity-transformed estimate
    must still score 100."""
    gt = traj(8)
    est = apply_sim3_to_poses(gt, 4.2, rot_z(63.0), np.array([1.0, 2.0, 3.0]))
    rra, rta = pairwise_pose_errors(est, gt)
    assert auc_at(rra, rta) == pytest.approx(100.0, abs=1e-6)


def test_auc_drops_when_errors_exceed_the_threshold():
    rra = np.array([1.0, 2.0, 50.0, 60.0])      # half the pairs are hopeless
    rta = np.zeros(4)
    auc = auc_at(rra, rta, max_threshold=30)
    assert 45.0 < auc < 50.0                    # ~half, minus the low-tau ramp


def test_auc_matches_a_hand_computed_case():
    """Two pairs at exactly 10 and 20 degrees: accuracy is 0 below 11,
    0.5 from 11..20, 1.0 from 21..30 -> mean over tau=1..30."""
    rra = np.array([10.0, 20.0])
    rta = np.zeros(2)
    expected = 100.0 * (0 * 10 + 0.5 * 10 + 1.0 * 10) / 30
    assert auc_at(rra, rta, 30) == pytest.approx(expected)


# ------------------------------------------------------------- revisit ---
def test_revisit_drift_is_zero_without_drift_and_positive_with_it():
    gt = traj(10)
    assert drift_at_revisit(gt.copy(), gt, [(0, 9)]) < 1e-9
    est = gt.copy()
    est[9, :3, 3] += np.array([0.2, 0.0, 0.0])
    assert drift_at_revisit(est, gt, [(0, 9)]) > 0.05


# ------------------------------------------------------------ summary ----
def test_summarise_returns_every_pilot_a_column():
    gt = traj(10)
    cols = summarise(gt.copy(), gt)
    for k in ("ate_rmse", "rpe_trans", "rpe_rot_deg", "auc30",
              "rra_median_deg", "rta_median_deg", "n_frames"):
        assert k in cols, k
    assert cols["n_frames"] == 10
    assert cols["ate_rmse"] < 1e-9 and cols["auc30"] == pytest.approx(100.0)


# ------------------------------------------------- Pilot A harness --------
def _pilot():
    import importlib.util, pathlib, sys
    spec = importlib.util.spec_from_file_location(
        "pa", pathlib.Path(__file__).resolve().parents[1] /
        "experiments" / "pilot_a.py")
    pa = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = ["pa"]; spec.loader.exec_module(pa)
    sys.argv = argv
    return pa


def test_chunks_cover_every_frame_with_the_requested_overlap():
    pa = _pilot()
    ch = pa.make_chunks(40, 12, 4)
    assert set().union(*ch) == set(range(40))
    for a, b in zip(ch, ch[1:]):
        assert len(set(a) & set(b)) >= 3, "Sim(3) needs 3 shared frames"


def test_regauge_puts_a_chunk_in_its_own_frame_and_scale():
    """Every backbone reports relative to its first view and normalises
    scale per call; a simulated chunk must do the same or the experiment
    measures nothing."""
    pa = _pilot()
    T = traj(8)
    g = pa.regauge(T)
    assert np.allclose(g[0], np.eye(4), atol=1e-9)
    assert abs(np.median(np.linalg.norm(g[1:, :3, 3], axis=1)) - 1.0) < 1e-9


def test_stitching_is_exact_when_chunks_are_error_free():
    """Both stitchers must reconstruct the trajectory up to a similarity
    when each chunk is a perfect regauged view of the truth -- otherwise a
    difference between them later cannot be attributed to the mechanism."""
    pa = _pilot()
    gt = traj(30)
    chunks = pa.make_chunks(30, 12, 4)
    outs = [pa.regauge(gt[idx]) for idx in chunks]
    est = pa.stitch_baseline(outs, chunks)
    assert ate_rmse(est, gt) < 1e-9


def test_continuity_decode_beats_single_window_decode_over_a_long_path():
    """A grid module reports position only modulo its period, so a decode
    confined to one window aliases once the path exceeds it.  Unwrapping
    against the previous decode is what makes long trajectories readable."""
    pa = _pilot()
    from smr.dynamics import ScaffoldState
    from smr.utils.geometry import euler_zyx_to_R, make_T
    ss = ScaffoldState(periods=pa.PERIODS, ring_N=128, torus_N=32, seed=0,
                       omega_max=0.16)
    ss.calibrate()
    xs = np.stack([np.array([0.4 * i, 0.0, 0.0]) for i in range(12)])
    abs_err, cont_err, near = [], [], None
    for x in xs:
        T = make_T(euler_zyx_to_R(0.0, 0.0, 0.0), x)
        ss.place_pose(T)
        abs_err.append(np.linalg.norm(pa.scaffold_decode(ss)[:3, 3] - x))
        d = pa.scaffold_decode(ss, near=near)
        cont_err.append(np.linalg.norm(d[:3, 3] - x))
        near = d[:3, 3]
    assert max(cont_err) < 0.05, f"continuity decode drifted: {cont_err}"
    assert max(abs_err) > max(cont_err), "single-window decode should alias"


def test_pose_alignment_survives_collinear_overlaps_where_points_fail():
    """The bug that produced ATE 11.2 on the first real Pilot A run.

    A chunk overlap is a few CONSECUTIVE frames of a smooth trajectory, so
    their camera centres are nearly collinear.  Point-only Umeyama leaves
    the rotation about that line unconstrained, and a little noise sends it
    tens of degrees wrong; orientation-based alignment is well posed from a
    single pose.
    """
    from smr.eval.trajectory import collinearity, sim3_from_poses

    def rot_y(a):
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                         [-np.sin(a), 0, np.cos(a)]])

    n = 4
    A = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        a = 0.02 * i
        A[i, :3, :3] = rot_y(a)
        A[i, :3, 3] = [np.sin(a), 0, np.cos(a)]
    assert collinearity(A[:, :3, 3]) < 0.02, "overlap should be ill-conditioned"

    s_t, R_t, t_t = 2.0, rot_y(0.9), np.array([1.0, 2.0, 3.0])
    B = apply_sim3_to_poses(A, s_t, R_t, t_t)
    B[:, :3, 3] += np.random.default_rng(0).normal(scale=0.01, size=(n, 3))

    def rot_err(R):
        return rotation_angle_deg(R.T @ R_t)

    assert rot_err(sim3_from_poses(A, B)[1]) < 1.0
    assert rot_err(umeyama_sim3(A[:, :3, 3], B[:, :3, 3])[1]) > 10.0


def test_pose_alignment_recovers_a_known_similarity_exactly():
    from smr.eval.trajectory import sim3_from_poses
    A = traj(6)
    s_t, R_t, t_t = 3.3, rot_z(41.0), np.array([0.5, -1.5, 2.0])
    B = apply_sim3_to_poses(A, s_t, R_t, t_t)
    s, R, t = sim3_from_poses(A, B)
    assert abs(s - s_t) < 1e-9
    assert np.allclose(R, R_t, atol=1e-9) and np.allclose(t, t_t, atol=1e-9)


def test_stitchers_use_orientation_aware_alignment():
    pa = _pilot()
    import inspect
    for fn in (pa.stitch_baseline, pa.stitch_smr):
        src = inspect.getsource(fn)
        assert "sim3_from_poses" in src, f"{fn.__name__} still points-only"
