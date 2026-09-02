"""Tests for smr.eval.mvs_fusion (no GPU).

1. Perfect multi-view depth of a plane is fully self-consistent; a 5 % depth
   corruption in one view is removed by the 1 % relative threshold; averaging
   returns the exact depth on clean data.
2. Trimmed Umeyama recovers a known Sim(3) under 20 % gross outliers; plain
   Umeyama does not.
"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from smr.eval.mvs_fusion import geo_consistency, select_sources, umeyama_trimmed, apply_sim3  # noqa: E402


def look_at(cam, target, up=(0, 1, 0)):
    """c2w with OpenCV axes (x right, y down, z forward)."""
    z = target - cam; z = z / np.linalg.norm(z)
    up = np.asarray(up, float)
    x = np.cross(z, up); x /= np.linalg.norm(x)          # right
    y = np.cross(z, x)                                    # down
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, cam
    return M


def plane_depth(K, c2w, H, W, z0=0.0):
    """Exact depth of the world plane z = z0 seen by a pinhole camera."""
    ys, xs = np.mgrid[0:H, 0:W]
    rays = np.linalg.inv(K) @ np.stack([xs.ravel(), ys.ravel(), np.ones(H * W)])
    d_w = c2w[:3, :3] @ rays                              # ray dirs in world
    o = c2w[:3, 3]
    lam = (z0 - o[2]) / d_w[2]                            # o_z + lam d_z = z0
    return lam.reshape(H, W)                              # depth = lambda (rays have z=1 in cam)


def make_scene(V=6, H=48, W=64, f=60.0):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    rng = np.random.default_rng(0)
    c2ws, depths = [], []
    for v in range(V):
        ang = 2 * np.pi * v / V
        cam = np.array([0.6 * np.cos(ang), 0.6 * np.sin(ang), 3.0 + 0.2 * rng.standard_normal()])
        c2w = look_at(cam, np.array([0.05 * np.cos(ang), 0.05 * np.sin(ang), 0.0]), up=(0, 1, 0.1))
        c2ws.append(c2w); depths.append(plane_depth(K, c2w, H, W))
    return np.stack(depths), np.stack([K] * V), np.stack(c2ws)


def test_clean_plane_is_consistent_and_average_is_exact():
    depths, Ks, c2ws = make_scene()
    src = select_sources(c2ws, 5)
    keep, dout, n = geo_consistency(depths, Ks, c2ws, src, pix_thr=1.0, rel_thr=0.01, min_views=5)
    # centre region is seen by every camera
    c = keep[:, 16:32, 24:40]
    assert c.mean() > 0.99, f"clean plane not consistent: {c.mean():.3f}"
    err = np.abs(dout[keep] - depths[keep]) / depths[keep]
    assert err.max() < 1e-4, f"averaging altered exact depth by {err.max():.2e}"  # bilinear source lookup


def test_corrupted_view_is_removed():
    depths, Ks, c2ws = make_scene()
    bad = depths.copy()
    bad[0, 10:38, 12:52] *= 1.05                          # 5 % error, far beyond 1 %
    src = select_sources(c2ws, 5)
    keep_clean, _, _ = geo_consistency(depths, Ks, c2ws, src, min_views=3)
    keep_bad, _, n = geo_consistency(bad, Ks, c2ws, src, min_views=3)
    region_clean = keep_clean[0, 10:38, 12:52].mean()
    region_bad = keep_bad[0, 10:38, 12:52].mean()
    assert region_clean > 0.95 and region_bad < 0.05, (region_clean, region_bad)
    # other views lose at most one supporting source (view 0), so with
    # min_views=3 they stay kept almost everywhere in the centre
    assert keep_bad[1:, 16:32, 24:40].mean() > 0.95


def test_trimmed_umeyama_beats_plain_under_outliers():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 3))
    s, ang = 1.7, 0.4
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    t = np.array([0.3, -1.2, 2.0])
    Y = apply_sim3(X, s, R, t) + 0.001 * rng.standard_normal(X.shape)
    out = rng.random(len(X)) < 0.2
    Y[out] += rng.standard_normal((out.sum(), 3)) * 3.0
    s0, R0, t0, _ = umeyama_trimmed(X, Y, trim=0.0)
    s1, R1, t1, inl = umeyama_trimmed(X, Y, trim=0.15, iters=4)
    e0 = np.linalg.norm(apply_sim3(X[~out], s0, R0, t0) - Y[~out], axis=1).mean()
    e1 = np.linalg.norm(apply_sim3(X[~out], s1, R1, t1) - Y[~out], axis=1).mean()
    assert e1 < 0.01 and e1 < 0.2 * e0, (e0, e1)
    assert abs(s1 - s) < 0.01


if __name__ == "__main__":
    test_clean_plane_is_consistent_and_average_is_exact()
    test_corrupted_view_is_removed()
    test_trimmed_umeyama_beats_plain_under_outliers()
    print("mvs_fusion tests passed")
