"""Long-horizon metrics (v65): loop-closure error, GT revisit pairs, the
within/cross-pass AUC split, scale drift, metric-distance RPE.  Each is
checked on a case whose value can be derived by hand."""
import numpy as np

from smr.eval.trajectory import (apply_sim3_to_poses, auc_split,
                                 find_revisit_pairs, loop_closure_error,
                                 rpe_at_distance, scale_drift, summarise_long)
from smr.stitch import sim3


def two_laps(n=20, laps=2, r=1.0):
    T = np.tile(np.eye(4), (n * laps, 1, 1))
    for k in range(n * laps):
        a = 2 * np.pi * (k % n) / n
        T[k, :3, :3] = sim3.rotmat([0, a, 0])
        T[k, :3, 3] = [r * np.cos(a), 0.0, r * np.sin(a)]
    return T


def test_find_revisit_pairs_on_two_laps():
    gt = two_laps(20, 2)
    pairs = find_revisit_pairs(gt, min_gap=10, max_dist=0.05, max_angle_deg=5)
    # every lap-one frame is revisited exactly one lap later
    assert pairs == [(i, i + 20) for i in range(20)]


def test_loop_closure_error_zero_under_similarity_and_reads_back_injected_rotation():
    gt = two_laps(20, 2)
    pairs = [(i, i + 20) for i in range(20)]
    est = apply_sim3_to_poses(gt, 2.0, sim3.rotmat([0.2, 0.1, 0.3]), np.array([1, 2, 3.0]))
    lc = loop_closure_error(est, gt, pairs)
    assert lc["rot_deg_mean"] < 1e-5 and lc["trans_mean"] < 1e-9 and lc["n"] == 20  # arccos floor ~1e-7 deg
    # rotate frame 25 by 7 degrees: exactly one pair (5, 25) sees it
    est2 = est.copy()
    est2[25, :3, :3] = sim3.rotmat([0, 0, np.radians(7.0)]) @ est2[25, :3, :3]
    lc2 = loop_closure_error(est2, gt, [(5, 25)])
    assert abs(lc2["rot_deg_mean"] - 7.0) < 1e-6
    # translate frame 25 by 0.1 (in a gauge with scale 2 -> 0.05 GT units)
    est3 = est.copy()
    est3[25, :3, 3] += 0.1 * est3[25, :3, :3][:, 0]
    lc3 = loop_closure_error(est3, gt, [(5, 25)], scale=0.5)
    assert abs(lc3["trans_mean"] - 0.05) < 1e-9


def test_auc_split_partitions_pairs_and_is_100_on_truth():
    gt = two_laps(12, 1)
    groups = [[0, 1, 2, 3], [2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    out = auc_split(gt, gt, groups)
    n = 12 * 11 // 2
    assert out["n_within"] + out["n_cross"] == n
    # within pairs: C(4,2)+C(4,2)+C(6,2) minus the shared pair (2,3)
    assert out["n_within"] == 6 + 6 + 15 - 1
    assert out["auc_within"] == 100.0 and out["auc_cross"] == 100.0


def test_scale_drift_zero_for_global_similarity_and_positive_for_ramp():
    gt = two_laps(20, 2)
    est = apply_sim3_to_poses(gt, 3.0, np.eye(3), np.zeros(3))
    assert scale_drift(est, gt, window=8)["max_abs_log"] < 1e-9
    ramp = gt.copy()
    for k in range(len(ramp)):
        ramp[k, :3, 3] *= 1.0 + 0.5 * k / len(ramp)
    assert scale_drift(ramp, gt, window=8)["max_abs_log"] > 0.05


def test_rpe_at_distance_exact_and_injected():
    gt = two_laps(20, 1, r=1.0)          # neighbours 2*sin(pi/20) ~ 0.313 apart
    t, r, n = rpe_at_distance(gt, gt, dist=0.5)
    assert t < 1e-9 and r < 1e-9 and n > 0
    est = gt.copy()
    est[:, :3, :3] = np.einsum("ij,kjl->kil", sim3.rotmat([0, 0, 0.0]), est[:, :3, :3])
    for k in range(len(est)):            # a 2-degree rotation per frame
        est[k, :3, :3] = sim3.rotmat([0, np.radians(2.0 * k), 0]) @ gt[k, :3, :3]
    _, r2, _ = rpe_at_distance(est, gt, dist=0.5)
    # 0.5 units of path = 2 frame steps -> 4 degrees of injected rotation
    assert abs(r2 - 4.0) < 0.3


def test_summarise_long_has_every_column():
    gt = two_laps(20, 2)
    cols = summarise_long(gt, gt, [list(range(0, 16)), list(range(8, 24)),
                                   list(range(16, 40))],
                          [(i, i + 20) for i in range(20)], chunk_len=16, rpe_dist=0.5)
    for k in ("ate_rmse", "rpe_trans_chunk", "rpe_rot_chunk_deg", "auc_all",
              "auc_within", "auc_cross", "loop_rot_deg", "loop_trans",
              "scale_drift_max", "rpe_trans_dist", "n_revisit_pairs"):
        assert k in cols
    assert cols["auc_all"] == 100.0 and cols["loop_rot_deg"] < 1e-5
