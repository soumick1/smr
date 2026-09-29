#!/usr/bin/env python3
"""VGGT vs COLMAP ground truth on a real LLFF scene.

    python scripts/llff_pose_check.py --scene lab_photos/room --res images_8

Loads poses_bounds.npy (COLMAP-registered), runs VGGT on the matching
image folder, and reports pose accuracy on REAL images.

Two metric layers, deliberately:
  ROBUST (convention-proof): centre errors after Sim(3) Umeyama alignment
    (world frame arbitrary) and relative-rotation MAGNITUDE errors over
    adjacent pairs (invariant to any fixed camera-axis permutation).
  CONVENTION-DEPENDENT (VERIFY): absolute rotation errors under the
    documented LLFF->OpenCV conversion below.  LLFF stores camera-to-world
    3x5 blocks [R | t | hwf] with camera axes [down, right, backward];
    ours are [right, down, forward], so R_ours = R_llff @ M with
    M = [[0,1,0],[1,0,0],[0,0,-1]] (det +1).  If the absolute numbers
    disagree wildly with the robust ones, suspect this conversion first.

Diagnostics printed for the GT itself (view-direction spread, centre
planarity) catch a wrong conversion: LLFF captures are forward-facing,
so GT view dirs must be nearly parallel.
"""
import argparse, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt                          # noqa: E402
from smr.backbones import get_backbone                   # noqa: E402
from smr.viz import TEAL, CORAL, GRAY, savefig           # noqa: E402

M_LLFF = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])


def load_llff(scene_dir, res):
    sc = pathlib.Path(scene_dir)
    arr = np.load(sc / "poses_bounds.npy")               # (N, 17)
    poses = arr[:, :15].reshape(-1, 3, 5)
    hwf = poses[0, :, 4]                                  # H, W, focal (orig)
    T = np.tile(np.eye(4), (len(poses), 1, 1))
    T[:, :3, :3] = poses[:, :, :3] @ M_LLFF               # VERIFY conversion
    T[:, :3, 3] = poses[:, :, 3]
    img_dir = sc / res
    paths = sorted(sum([[str(p) for p in img_dir.glob(pat)]
                        for pat in ("*.png", "*.jpg", "*.JPG", "*.jpeg")], []))
    assert len(paths) == len(poses), \
        f"{len(paths)} images vs {len(poses)} poses -- wrong --res folder?"
    return T, hwf, paths


def umeyama_sim3(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (Xc ** 2).sum() * len(X)
    return s, R, my - s * R @ mx


def rot_angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--res", default="images_8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vggt")
    a = ap.parse_args()
    T_gt, hwf, paths = load_llff(a.scene, a.res)
    name = pathlib.Path(a.scene).name

    # GT self-diagnostics (catch a broken conversion before blaming VGGT)
    z = T_gt[:, :3, 2]                                   # our forward axes
    spread = np.degrees(np.arccos(np.clip(z @ z.mean(0)
                                          / np.linalg.norm(z.mean(0)), -1, 1)))
    print(f"  GT diag [{name}]: view-dir spread mean {spread.mean():.1f} deg "
          f"(forward-facing capture: expect small, ~<25)")

    bb = get_backbone(a.backbone, device=a.device)
    out = bb.infer(paths)

    C_est, C_gt = out.poses[:, :3, 3], T_gt[:, :3, 3]
    s, R, t = umeyama_sim3(C_est, C_gt)
    scene_scale = np.linalg.norm(C_gt.max(0) - C_gt.min(0))
    cen = [np.linalg.norm(s * R @ C_est[i] + t - C_gt[i]) / scene_scale
           for i in range(len(paths))]
    # convention-robust relative-rotation magnitudes (adjacent pairs)
    rel = []
    for i in range(len(paths) - 1):
        a_e = rot_angle(out.poses[i, :3, :3].T @ out.poses[i + 1, :3, :3])
        a_g = rot_angle(T_gt[i, :3, :3].T @ T_gt[i + 1, :3, :3])
        rel.append(abs(a_e - a_g))
    # convention-dependent absolute rotation (VERIFY block above)
    rot = [rot_angle((R @ out.poses[i, :3, :3]).T @ T_gt[i, :3, :3])
           for i in range(len(paths))]
    print(f"  [ROBUST] centre err / scene scale: median "
          f"{np.median(cen):.4f}, mean {np.mean(cen):.4f} (soft thr 0.05)")
    print(f"  [ROBUST] relative-rotation magnitude err (adjacent): median "
          f"{np.median(rel):.3f} deg (soft thr 1.0)")
    print(f"  [VERIFY-CONVENTION] absolute rotation err: median "
          f"{np.median(rot):.2f} deg, mean {np.mean(rot):.2f}")

    Ca = (s * (R @ C_est.T)).T + t
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.0))
    axs[0].imshow(np.clip(out.rgb[0], 0, 1)); axs[0].axis("off")
    axs[0].set_title(f"{name} view 0")
    axs[1].plot(C_gt[:, 0], C_gt[:, 2], "o-", color=GRAY, label="COLMAP GT")
    axs[1].plot(Ca[:, 0], Ca[:, 2], "x--", color=TEAL,
                label=f"{a.backbone} aligned")
    axs[1].legend(); axs[1].axis("equal"); axs[1].set_title("centres (top-down)")
    axs[2].bar(range(len(cen)), cen, color=CORAL)
    axs[2].set_title("centre err / scene scale")
    savefig(fig, pathlib.Path(f"outputs/figures/llff_{a.backbone}_{name}.png"))
    import json
    pathlib.Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"outputs/reports/llff_{a.backbone}_{name}.json").write_text(
        json.dumps(dict(centre_median=float(np.median(cen)),
                        centre_mean=float(np.mean(cen)),
                        relrot_median_deg=float(np.median(rel)),
                        absrot_median_deg=float(np.median(rot))), indent=2))


if __name__ == "__main__":
    main()
