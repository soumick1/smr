"""ETH3D GT builders recover a known tilted plane (no data download needed).

Constructs a THIN_PRISM_FISHEYE 'distorted' camera + PINHOLE 'undistorted'
camera sharing a pose, renders the plane into a raw float32 depth file in
the official layout, and checks that gt_depth_official -> unproject_grid
lands on the plane (|z| small).  Same for the laser-scan z-buffer fallback.
"""
import sys, pathlib, tempfile
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from smr.eval.eth3d_gt import (gt_depth_official, gt_depth_from_scan, unproject_grid,  # noqa: E402
                               fisheye_project)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def make_camera():
    Wd, Hd = 640, 480                       # 'distorted' original
    fx = fy = 500.0; cx, cy = 320.0, 240.0
    dist = np.array([0.05, 0.01, 0.0005, -0.0003, 0.0, 0.0, 0.0001, -0.0001])
    Wu, Hu = 640, 480                       # undistorted grid (same size for the test)
    K_und = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    c2w = np.eye(4); c2w[:3, :3] = rot_x(np.deg2rad(200.0)); c2w[:3, 3] = [0.2, 0.1, 3.0]
    # camera looks roughly down (-z world) with a 20 deg tilt
    return dict(Wd=Wd, Hd=Hd, fx=fx, fy=fy, cx=cx, cy=cy, dist=dist, Wu=Wu, Hu=Hu, K_und=K_und, c2w=c2w)


def plane_depth_for_rays(x, y, c2w, z0=0.0):
    """Depth (camera z) at which the ray (x, y, 1) hits world plane z = z0."""
    d = np.stack([x, y, np.ones_like(x)], -1) @ c2w[:3, :3].T        # ray dirs in world
    lam = (z0 - c2w[2, 3]) / d[..., 2]
    return lam


def write_scene(tmp, cam):
    scene = pathlib.Path(tmp) / "scene"
    (scene / "dslr_calibration_jpg").mkdir(parents=True)
    (scene / "ground_truth_depth" / "dslr_images").mkdir(parents=True)
    p = cam["dist"]
    (scene / "dslr_calibration_jpg" / "cameras.txt").write_text(
        "# cams\n0 THIN_PRISM_FISHEYE %d %d %g %g %g %g %s\n" % (
            cam["Wd"], cam["Hd"], cam["fx"], cam["fy"], cam["cx"], cam["cy"], " ".join(f"{v:g}" for v in p)))
    (scene / "dslr_calibration_jpg" / "images.txt").write_text(
        "# imgs\n1 1 0 0 0 0 0 0 0 dslr_images/IMG.JPG\n\n")
    # render the plane into the DISTORTED image by forward-scattering a 3x oversampled undistorted grid
    us = (np.arange(3 * cam["Wu"]) + 0.5) / 3.0; vs = (np.arange(3 * cam["Hu"]) + 0.5) / 3.0
    gx, gy = np.meshgrid((us - cam["cx"]) / cam["fx"], (vs - cam["cy"]) / cam["fy"])
    depth = plane_depth_for_rays(gx, gy, cam["c2w"])
    dx, dy = fisheye_project(gx, gy, cam["dist"])
    px = np.floor(dx * cam["fx"] + cam["cx"]).astype(int); py = np.floor(dy * cam["fy"] + cam["cy"]).astype(int)
    ok = (px >= 0) & (px < cam["Wd"]) & (py >= 0) & (py < cam["Hd"]) & (depth > 0)
    D = np.full((cam["Hd"], cam["Wd"]), np.inf, np.float32)
    np.minimum.at(D, (py[ok], px[ok]), depth[ok].astype(np.float32))
    D[:, :40] = np.inf                                             # an 'official mask' stripe
    D.astype("<f4").tofile(scene / "ground_truth_depth" / "dslr_images" / "IMG.JPG")
    return scene


def test_official_depth_path_recovers_plane():
    cam = make_camera()
    with tempfile.TemporaryDirectory() as tmp:
        scene = write_scene(tmp, cam)
        depth, st = gt_depth_official(scene, "IMG.JPG", cam["K_und"], (cam["Wu"], cam["Hu"]), (60, 80))
    assert 0.5 < st["grid_valid"] < 1.0, st                       # mask stripe removed some cells
    # barrel compression maps only the outermost undistorted column into the
    # 40-px masked stripe: column 0 must carry NaNs, the interior none
    assert np.isnan(depth[:, 0]).mean() > 0.3 and np.isfinite(depth[:, 2:]).all()
    P = unproject_grid(depth, cam["K_und"], cam["c2w"], (cam["Wu"], cam["Hu"]))
    z = P[..., 2][np.isfinite(P[..., 2])]
    assert np.abs(z).max() < 0.02, f"plane not recovered: max |z| = {np.abs(z).max():.4f}"


def test_scan_zbuffer_path_recovers_plane():
    cam = make_camera()
    rng = np.random.default_rng(0)
    scan = np.stack([rng.uniform(-3, 3, 400000), rng.uniform(-3, 3, 400000), np.zeros(400000)], -1)
    depth = gt_depth_from_scan(scan, cam["K_und"], cam["c2w"], (cam["Wu"], cam["Hu"]), (60, 80))
    assert np.isfinite(depth).mean() > 0.9
    P = unproject_grid(depth, cam["K_und"], cam["c2w"], (cam["Wu"], cam["Hu"]), stride=2)
    z = P[..., 2][np.isfinite(P[..., 2])]
    # nearest-depth z-buffer per 8x8-px cell on a 20-deg slanted plane at 3 m:
    # footprint 4.8 cm x tan(20 deg) -> up to ~1.7 cm bias, the known resolution limit
    assert np.abs(z).max() < 0.02, np.abs(z).max()
    assert P.shape == (30, 40, 3)


if __name__ == "__main__":
    test_official_depth_path_recovers_plane()
    test_scan_zbuffer_path_recovers_plane()
    print("eth3d_gt tests passed")
