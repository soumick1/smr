"""v133: native point maps through assemble() and the confidence guard (no GPU)."""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
from smr.backbones.pointmap import RawViews, assemble
import points_suite as PS


def test_assemble_world_points_match_poses_and_depth():
    rng = np.random.default_rng(0)
    K_, H, W = 3, 12, 16
    gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    pts_local = np.stack([np.stack([gx, gy, np.full((H, W), 4.0 + k)], -1) for k in range(K_)])   # (K,H,W,3), z = 4+k
    poses = np.tile(np.eye(4), (K_, 1, 1)); poses[:, :3, 3] = rng.normal(0, 1, (K_, 3))
    raw = RawViews(poses=poses, rgb=np.zeros((K_, H, W, 3)), conf=np.ones((K_, H, W)) * 3.0, pts_local=pts_local)
    out = assemble(raw, "c2w")
    wp = out.extras["world_points"]; s = out.extras["scene_scale"]
    assert wp.shape == (K_, H, W, 3)
    # world point = pose @ (local / s); poses in `out` already have t / s
    exp = pts_local / s + out.poses[:, None, None, :3, 3]
    assert np.allclose(wp, exp), np.abs(wp - exp).max()
    assert np.allclose(wp[..., 2] - out.poses[:, None, None, 2, 3], out.depth)   # z consistent with depth


def test_conf_guard_falls_back_to_percentile_for_sigmoid_conf():
    class O: pass
    o = O(); rng = np.random.default_rng(1)
    o.extras = dict(conf=rng.random((2, 20, 30)))                      # sigmoid-like, all < 1
    m = PS.conf_mask(o, 0, None, abs_thr=2.0)
    assert 0.6 < m.mean() < 0.75, m.mean()                             # top 68% by percentile, not empty
    o.extras = dict(conf=1.0 + np.exp(rng.normal(0, 1, (2, 20, 30))))    # expp1-like: absolute cut applies
    m2 = PS.conf_mask(o, 0, None, abs_thr=2.0)
    assert abs(m2.mean() - (o.extras["conf"][0] > 2.0).mean()) < 1e-9


if __name__ == "__main__":
    test_assemble_world_points_match_poses_and_depth()
    test_conf_guard_falls_back_to_percentile_for_sigmoid_conf()
    print("native pointmap tests passed")
