import numpy as np

from smr.stitch import sim3
from smr.stitch.consensus_pose import auc_at, consensus_poses, pairwise_errors


def _random_poses(rng, n):
    P = []
    for i in range(n):
        T = np.eye(4)
        T[:3, :3] = sim3.rotmat(rng.normal(scale=0.4, size=3))
        T[:3, 3] = rng.uniform(-2, 2, 3)
        P.append(T)
    return np.stack(P)


def _noisy_pass(rng, gt, ids, rot_sig, t_sig, gauge_scale):
    S = (gauge_scale, sim3.rotmat(rng.normal(scale=0.5, size=3)), rng.uniform(-1, 1, 3))
    P = []
    for g in ids:
        T = sim3.apply_one(S, gt[g])
        T[:3, :3] = sim3.rotmat(rng.normal(scale=np.radians(rot_sig), size=3)) @ T[:3, :3]
        T[:3, 3] += rng.normal(scale=t_sig, size=3)
        P.append(T)
    return np.stack(P)


def test_consensus_over_contexts_beats_one_pass():
    rng = np.random.default_rng(0)
    n, k = 10, 8
    gains = []
    for trial in range(6):
        gt = _random_poses(rng, n)
        single = _noisy_pass(rng, gt, list(range(n)), rot_sig=3.0, t_sig=0.05, gauge_scale=1.0)
        passes = [(list(range(n)), single)]
        for _ in range(k - 1):
            ids = list(rng.permutation(n))
            passes.append((ids, _noisy_pass(rng, gt, ids, 3.0, 0.05, rng.uniform(0.5, 2.0))))
        fused, ctx = consensus_poses(passes, n)
        r1, t1 = pairwise_errors(single, gt)
        rf, tf = pairwise_errors(fused, gt)
        gains.append(auc_at(rf, tf) - auc_at(r1, t1))
        assert min(ctx.values()) >= 2
    assert np.mean(gains) > 0.03, gains          # consensus must clearly beat one context


def test_consensus_is_noop_for_deterministic_backbone():
    rng = np.random.default_rng(1)
    n = 10
    gt = _random_poses(rng, n)
    exact = gt.copy()
    passes = [(list(range(n)), exact.copy()) for _ in range(5)]
    fused, _ = consensus_poses(passes, n)
    rf, tf = pairwise_errors(fused, gt)
    assert auc_at(rf, tf) > 0.999                 # permutation-equivariant case: no harm


def test_pairwise_median_survives_bad_reference_passes():
    """Half the passes are globally poor (bad reference frame); per-pair
    medians must track the clean regime, where global averaging degrades."""
    from smr.stitch.consensus_pose import pairwise_median_errors
    rng = np.random.default_rng(2)
    n = 10
    for trial in range(4):
        gt = _random_poses(rng, n)
        passes = []
        for k in range(8):
            sig = 2.0 if k % 2 == 0 else 10.0          # alternating good/bad passes
            ids = list(rng.permutation(n)) if k else list(range(n))
            passes.append((ids, _noisy_pass(rng, gt, ids, sig, 0.02 * sig, rng.uniform(0.5, 2))))
        r1, t1 = pairwise_errors(passes[0][1], gt)      # the single clean pass
        rm, tm = pairwise_median_errors(passes, gt)
        assert auc_at(rm, tm) >= auc_at(r1, t1) - 0.02  # never meaningfully worse than single
