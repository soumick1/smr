"""7-Scenes / TUM ground-truth loaders on synthetic fixtures written to
tmp_path: frame ordering, pose parsing, invalid-pose dropping, timestamp
association, quaternion conversion, and the convention-variant table."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "indoor_gt_poses", ROOT / "scripts" / "indoor_gt_poses.py")
igp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(igp)


def _png(path):
    try:
        from PIL import Image
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)
    except ImportError:                                       # pragma: no cover
        import imageio.v2 as iio
        iio.imwrite(path, np.zeros((8, 8, 3), np.uint8))


def _rot(deg, axis="z"):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_sevenscenes_loader_orders_frames_and_drops_invalid(tmp_path):
    d = tmp_path / "chess" / "seq-01"
    d.mkdir(parents=True)
    truth = {}
    for fid in (0, 1, 2, 3):
        P = np.eye(4)
        P[:3, :3] = _rot(10 * fid)
        P[:3, 3] = [fid * 0.1, 0.0, 1.0]
        truth[fid] = P
        np.savetxt(d / f"frame-{fid:06d}.pose.txt", P, fmt="%.8e", delimiter="\t")
        _png(d / f"frame-{fid:06d}.color.png")
    # an invalid frame (7-Scenes ships a few of these)
    np.savetxt(d / "frame-000004.pose.txt", np.full((4, 4), -np.inf), fmt="%e")
    _png(d / "frame-000004.color.png")
    poses, paths, ids, dropped = igp.load_sevenscenes(tmp_path, "chess", 1)
    assert (ids - 100000).tolist() == [0, 1, 2, 3] and len(dropped) == 1
    assert dropped[0][1] == "invalid pose"
    for i, fid in enumerate(ids - 100000):
        assert np.allclose(poses[i], truth[fid])
        assert paths[i].endswith(f"frame-{fid:06d}.color.png")


def test_quaternion_conversion_known_rotations():
    assert np.allclose(igp.quat_to_R(0, 0, 0, 1), np.eye(3))
    s = np.sin(np.pi / 4)
    assert np.allclose(igp.quat_to_R(0, 0, s, s), _rot(90, "z"))
    assert np.allclose(igp.quat_to_R(s, 0, 0, s), _rot(90, "x"))
    # scaling the quaternion must not change the rotation
    assert np.allclose(igp.quat_to_R(0, 0, 2 * s, 2 * s), _rot(90, "z"))


def test_tum_association_and_poses(tmp_path):
    d = tmp_path / "rgbd_dataset_freiburg1_desk"
    (d / "rgb").mkdir(parents=True)
    rgb_lines = ["# rgb"]
    for t in (1.000, 1.033, 1.066, 5.000):        # the last has no GT nearby
        rgb_lines.append(f"{t:.6f} rgb/{t:.6f}.png")
        _png(d / "rgb" / f"{t:.6f}.png")
    (d / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    s = np.sin(np.pi / 4)
    gt_lines = ["# gt",
                "0.990 0 0 0 0 0 0 1",             # nearest to 1.000 (0.010 s)
                "1.040 1 2 3 0 0 %f %f" % (s, s),  # nearest to 1.033 (0.007 s)
                "1.060 4 5 6 0 0 0 1"]             # nearest to 1.066 (0.006 s)
    (d / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n")
    poses, paths, ids, dropped = igp.load_tum(tmp_path, d.name, max_dt=0.02)
    assert len(poses) == 3 and len(dropped) == 1
    assert np.allclose(poses[0], np.eye(4))
    assert np.allclose(poses[1][:3, :3], _rot(90, "z")) and np.allclose(poses[1][:3, 3], [1, 2, 3])
    assert np.allclose(poses[2][:3, 3], [4, 5, 6])
    assert paths[1].endswith("rgb/1.033000.png")


def test_pose_variants_cover_inverse_and_flips():
    P = np.eye(4)
    P[:3, :3] = _rot(30)
    P[:3, 3] = [1, 2, 3]
    v = igp.pose_variants(P)
    assert set(v) == {"c2w", "c2w_xy", "c2w_yz", "c2w_xz",
                      "w2c", "w2c_xy", "w2c_yz", "w2c_xz"}
    assert np.allclose(v["c2w"], P)
    assert np.allclose(v["w2c"], np.linalg.inv(P))
    assert np.allclose(v["c2w_xy"][:3, :3], P[:3, :3] @ np.diag([-1, -1, 1]))
    for name, V in v.items():
        assert abs(np.linalg.det(V[:3, :3]) - 1) < 1e-9, name


def test_sevenscenes_concatenates_sessions_with_unique_ids(tmp_path):
    for s in (1, 2):
        d = tmp_path / "chess" / f"seq-{s:02d}"
        d.mkdir(parents=True)
        for fid in range(3):
            P = np.eye(4)
            P[:3, 3] = [s, fid, 0]
            np.savetxt(d / f"frame-{fid:06d}.pose.txt", P, fmt="%.8e", delimiter="\t")
            _png(d / f"frame-{fid:06d}.color.png")
    poses, paths, ids, dropped = igp.load_sevenscenes(tmp_path, "chess", [1, 2])
    assert len(poses) == 6 and len(set(ids.tolist())) == 6
    assert np.allclose(poses[:, 0, 3], [1, 1, 1, 2, 2, 2])
    assert paths[3].endswith("seq-02/frame-000000.color.png")
