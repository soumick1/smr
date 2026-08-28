import numpy as np

from smr.stitch.imagine import depth_absrel, psnr, score_view, splat_points, ssim


def _pinhole(H=60, W=80, f=70.0):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return K, (H, W)


def test_splat_roundtrip_reproduces_geometry():
    rng = np.random.default_rng(0)
    K, hw = _pinhole()
    # a fronto-parallel plane at z = 3 must come back at exactly z = 3
    g = np.stack(np.meshgrid(np.linspace(-1.5, 1.5, 80), np.linspace(-1.2, 1.2, 60)), -1).reshape(-1, 2)
    plane = np.concatenate([g, np.full((len(g), 1), 3.0)], 1)
    imgp, depp, maskp = splat_points(plane, np.ones((len(plane), 3)) * 0.5, np.eye(4), K, hw)
    assert maskp.mean() > 0.5 and abs(np.median(depp[maskp]) - 3.0) < 1e-6
    # a random cloud: nearer points must win the z-buffer
    pts = rng.uniform([-1, -1, 2.0], [1, 1, 4.0], size=(4000, 3))
    col = rng.random((4000, 3))
    T = np.eye(4)
    img, dep, mask = splat_points(pts, col, T, K, hw)
    assert 0.3 < mask.mean() <= 1.0
    assert np.median(dep[mask]) <= np.median(pts[:, 2])
    # a second camera translated right sees the cloud shifted left
    T2 = np.eye(4); T2[0, 3] = 0.5
    img2, dep2, mask2 = splat_points(pts, col, T2, K, hw)
    assert mask2.mean() > 0.2
    c1 = np.array(np.nonzero(mask)).mean(1)
    c2 = np.array(np.nonzero(mask2)).mean(1)
    assert c2[1] < c1[1]                          # centroid moved left in pixels


def test_metrics_behave():
    rng = np.random.default_rng(1)
    a = rng.random((40, 50, 3))
    assert psnr(a, a) > 80 and abs(ssim(a, a) - 1) < 1e-6
    b = np.clip(a + rng.normal(scale=0.1, size=a.shape), 0, 1)
    assert 15 < psnr(a, b) < 25 and ssim(a, b) < 1
    d = rng.uniform(1, 3, (40, 50))
    assert depth_absrel(d * 1.1, d, np.ones_like(d, bool)) - 0.1 < 1e-6
    s = score_view(b, d * 1.1, np.ones_like(d, bool), a, d)
    assert set(s) >= {"psnr", "ssim", "coverage", "absrel"} and s["coverage"] == 1.0
