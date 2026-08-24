"""Sim(3) helper tests: every operation against a hand-derivable case."""
import numpy as np

from smr.eval.trajectory import apply_sim3_to_poses
from smr.stitch import sim3


def _rand_sim(rng, s=None):
    return (s if s is not None else float(np.exp(rng.normal(scale=0.3))),
            sim3.rotmat(rng.normal(size=3) * 0.4), rng.normal(size=3))


def _apply_pts(S, x):
    s, R, t = S
    return (s * (R @ x.T)).T + t


def test_compose_and_inverse_are_consistent():
    rng = np.random.default_rng(0)
    S1, S2 = _rand_sim(rng), _rand_sim(rng)
    x = rng.normal(size=(6, 3))
    assert np.allclose(_apply_pts(sim3.compose(S2, S1), x),
                       _apply_pts(S2, _apply_pts(S1, x)))
    assert np.allclose(_apply_pts(sim3.inverse(S1), _apply_pts(S1, x)), x)


def test_vector_parametrisation_round_trips():
    rng = np.random.default_rng(1)
    S = _rand_sim(rng)
    S2 = sim3.from_vec(sim3.to_vec(S))
    assert abs(S2[0] - S[0]) < 1e-12 and np.allclose(S2[1], S[1]) \
        and np.allclose(S2[2], S[2])


def test_rotvec_handles_small_and_near_pi_angles():
    assert np.allclose(sim3.rotvec(np.eye(3)), 0)
    R = sim3.rotmat([0, 0, np.pi - 1e-7])
    v = sim3.rotvec(R)
    assert abs(np.linalg.norm(v) - (np.pi - 1e-7)) < 1e-5
    assert np.allclose(sim3.rotmat(v), R, atol=1e-6)


def test_interpolate_is_identity_at_0_and_S_at_1():
    rng = np.random.default_rng(2)
    S = _rand_sim(rng)
    I = sim3.interpolate(S, 0.0)
    assert I[0] == 1.0 and np.allclose(I[1], np.eye(3)) and np.allclose(I[2], 0)
    F = sim3.interpolate(S, 1.0)
    assert abs(F[0] - S[0]) < 1e-12 and np.allclose(F[1], S[1]) and np.allclose(F[2], S[2])
    # half-way rotation is half the angle about the same axis
    H = sim3.interpolate(S, 0.5)
    assert abs(np.linalg.norm(sim3.rotvec(H[1])) - 0.5 * np.linalg.norm(sim3.rotvec(S[1]))) < 1e-9


def test_fit_poses_recovers_similarity_from_two_poses():
    """Two poses fix rotation (from orientations), scale (centre distance)
    and translation -- the minimum an anchor SITE must provide."""
    rng = np.random.default_rng(3)
    S = _rand_sim(rng, s=1.7)
    T = np.tile(np.eye(4), (2, 1, 1))
    T[1, :3, 3] = [1.0, 0.2, 0.0]
    T[1, :3, :3] = sim3.rotmat([0, 0.3, 0])
    B = apply_sim3_to_poses(T, *S)
    Sf, info = sim3.fit_poses(T, B)
    assert info["scale_ok"]
    assert abs(Sf[0] - S[0]) < 1e-9 and np.allclose(Sf[1], S[1]) and np.allclose(Sf[2], S[2])


def test_fit_poses_single_pose_leaves_scale_unresolved():
    T = np.eye(4)[None]
    Sf, info = sim3.fit_poses(T, T)
    assert not info["scale_ok"] and Sf[0] == 1.0


def test_robust_fit_rejects_a_wrong_correspondence():
    """One false anchor (a wrong loop closure) must not move the fit."""
    rng = np.random.default_rng(4)
    S = _rand_sim(rng)
    T = np.tile(np.eye(4), (7, 1, 1))
    for i in range(7):
        T[i, :3, 3] = rng.normal(size=3)
        T[i, :3, :3] = sim3.rotmat(rng.normal(size=3) * 0.5)
    B = apply_sim3_to_poses(T, *S)
    B[3, :3, :3] = sim3.rotmat([1.0, 0.3, 0.0]) @ B[3, :3, :3]
    B[3, :3, 3] += [3.0, -2.0, 1.0]
    Sf, keep, info = sim3.fit_poses_robust(T, B)
    assert keep.tolist() == [True, True, True, False, True, True, True]
    assert abs(Sf[0] - S[0]) < 1e-9 and np.allclose(Sf[1], S[1]) and np.allclose(Sf[2], S[2])
