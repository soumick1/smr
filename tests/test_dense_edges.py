import numpy as np

from smr.stitch import sim3
from smr.stitch.dense_edges import remeasure


class RV:
    pass


def _make_world(rng, n_frames, K, hw):
    H, W = hw
    ys, xs = np.mgrid[0:H:6, 0:W:6]
    gt = []
    for i in range(n_frames):
        T = np.eye(4); T[:3, 3] = [0.3 * i, 0, 0]
        gt.append(T)
    depth = rng.uniform(2.0, 4.0, (n_frames, H, W))
    return np.stack(gt), depth, (ys, xs)


def test_remeasure_seq_edge_recovers_relative_sim3():
    rng = np.random.default_rng(0)
    K = np.array([[80.0, 0, 48.0], [0, 80.0, 36.0], [0, 0, 1.0]])
    hw = (72, 96)
    gt, depth, _ = _make_world(rng, 12, K, hw)
    chunks = [list(range(8)), list(range(4, 12))]
    S1 = (1.0, np.eye(3), np.zeros(3))
    S2 = (1.5, sim3.rotmat([0.1, -0.2, 0.4]), np.array([1.0, -0.5, 0.3]))
    world_of = {0: S1, 1: S2}

    def infer_fn(paths):
        ids = [int(p) for p in paths]
        ci = 0 if set(ids) <= set(chunks[0]) else 1
        S = world_of[ci]
        rv = RV()
        rv.poses = np.stack([sim3.apply_one(sim3.inverse(S), gt[g]) for g in ids])
        D = np.zeros((len(ids), *hw))
        for li, g in enumerate(ids):
            D[li] = depth[g] / S[0]          # depth scales with the inverse gauge
        rv.depth = D
        rv.intrinsics = K
        return rv

    res = dict(edges=[dict(c=0, k=1, Z=sim3.compose(sim3.inverse(S1), S2), w_scale=1.0,
                           kind="seq")],
               est=np.stack([np.eye(4)] * 12),
               local={g: np.eye(4) for g in range(12)})
    # perturb the pose-fit Z so only the dense fit can restore it
    Zb = list(res["edges"][0]["Z"]); Zb[0] *= 1.2
    res["edges"][0]["Z"] = (Zb[0], Zb[1], Zb[2] + np.array([0.3, 0, 0]))
    out = remeasure(res, chunks, [str(g) for g in range(12)], infer_fn)
    Z = out["edges"][0]["Z"]
    Zt = sim3.compose(sim3.inverse(S1), S2)
    assert abs(Z[0] - Zt[0]) < 0.03
    assert np.linalg.norm(Z[2] - Zt[2]) < 0.1
