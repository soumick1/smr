#!/usr/bin/env python3
"""Ground-truth poses for indoor walkthrough datasets (7-Scenes, TUM RGB-D)
in the npz schema Pilot A consumes, with a measured convention check.

Why these datasets: chunking is the NATURAL regime for a room walkthrough
(distant frames share nothing, so the sequence cannot be ingested at once)
and both contain genuine revisits.  KV-Tracker (CVPR 2026) reports
full-sequence ATE on exactly these sequences, which gives Table 1 external
rows for free.

7-Scenes   <root>/<scene>/seq-XX/frame-NNNNNN.color.png + .pose.txt
           pose.txt is a 4x4 camera-to-world matrix.  A few frames carry
           non-finite entries; they are dropped and counted.
TUM RGB-D  <root>/rgbd_dataset_freiburgN_<name>/rgb.txt + groundtruth.txt
           groundtruth rows: t tx ty tz qx qy qz qw (camera-to-world).
           RGB timestamps are associated to the nearest GT timestamp within
           --max-dt (0.02 s, the TUM tools' default).

Both are documented as camera-to-world in an OpenCV-style camera frame,
which is what the backbones emit after the socket's conversion -- but a
convention is never taken on trust here: --diagnose-convention runs a real
backbone on frames spread over the sequence and scores every plausible
reading (as-is / inverted, with each axis flip).  The correct one wins by
a mile; a wrong one leaves camera CENTRES right and every orientation off.

    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes \\
        --scene chess --seq 1 --out data/gt/7scenes_chess_seq01.npz
    python scripts/indoor_gt_poses.py sevenscenes --root ~/7scenes \\
        --scene chess --seq 1 --diagnose-convention --backbone vggt
    python scripts/indoor_gt_poses.py tum --root ~/tum \\
        --sequence rgbd_dataset_freiburg1_desk --out data/gt/tum_fr1_desk.npz
"""
import argparse, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ------------------------------------------------------------- 7-Scenes --
def load_sevenscenes(root, scene, seq):
    """seq may be an int or a list of ints: sequences of one scene share a
    single world frame (KinectFusion tracked them in one scene model), so
    concatenating them yields a long multi-session trajectory whose
    cross-session revisits carry ground truth.  Frame ids are offset by
    100000 x seq so they stay unique."""
    if isinstance(seq, (list, tuple)):
        parts = [load_sevenscenes(root, scene, s) for s in seq]
        return (np.concatenate([p[0] for p in parts]),
                sum([p[1] for p in parts], []),
                np.concatenate([p[2] for p in parts]),
                sum([p[3] for p in parts], []))
    d = pathlib.Path(root).expanduser() / scene / f"seq-{int(seq):02d}"
    if not d.is_dir():
        raise SystemExit(f"{d} not found (expected <root>/<scene>/seq-XX)")
    poses, paths, ids, dropped = [], [], [], []
    for pose_file in sorted(d.glob("frame-*.pose.txt")):
        fid = int(pose_file.name.split("-")[1].split(".")[0])
        img = pose_file.with_name(f"frame-{fid:06d}.color.png")
        if not img.exists():
            dropped.append((fid, "no image"))
            continue
        try:
            P = np.loadtxt(pose_file, dtype=float)
        except ValueError:
            dropped.append((fid, "unparsable pose"))
            continue
        if P.shape != (4, 4) or not np.all(np.isfinite(P)) or \
                abs(np.linalg.det(P[:3, :3]) - 1.0) > 1e-2:
            dropped.append((fid, "invalid pose"))
            continue
        poses.append(P); paths.append(str(img)); ids.append(fid + 100000 * int(seq))
    if not poses:
        raise SystemExit(f"no valid frames under {d}")
    return np.stack(poses), paths, np.array(ids), dropped


# ------------------------------------------------------------------ TUM --
def quat_to_R(qx, qy, qz, qw):
    """Unit quaternion (x, y, z, w) -> rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def _read_list(path):
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line.split())
    return rows


def load_tum(root, sequence, max_dt=0.02):
    d = pathlib.Path(root).expanduser() / sequence
    if not d.is_dir():
        raise SystemExit(f"{d} not found")
    rgb = _read_list(d / "rgb.txt")
    gt = _read_list(d / "groundtruth.txt")
    t_gt = np.array([float(r[0]) for r in gt])
    poses, paths, ids, dropped = [], [], [], []
    order = np.argsort(t_gt)
    t_sorted = t_gt[order]
    for r in rgb:
        t = float(r[0])
        j = int(np.searchsorted(t_sorted, t))
        cand = [c for c in (j - 1, j) if 0 <= c < len(t_sorted)]
        if not cand:
            dropped.append((t, "no gt")); continue
        c = min(cand, key=lambda c: abs(t_sorted[c] - t))
        if abs(t_sorted[c] - t) > max_dt:
            dropped.append((t, f"nearest gt {abs(t_sorted[c] - t):.3f}s away"))
            continue
        row = gt[order[c]]
        tx, ty, tz, qx, qy, qz, qw = (float(v) for v in row[1:8])
        P = np.eye(4)
        P[:3, :3] = quat_to_R(qx, qy, qz, qw)
        P[:3, 3] = [tx, ty, tz]
        poses.append(P); paths.append(str(d / r[1])); ids.append(t)
    if not poses:
        raise SystemExit("no frames could be associated with ground truth")
    return np.stack(poses), paths, np.array(ids), dropped


# ------------------------------------------------------- convention check --
FLIPS = {"": np.eye(3), "_xy": np.diag([-1.0, -1.0, 1.0]),
         "_yz": np.diag([1.0, -1.0, -1.0]), "_xz": np.diag([-1.0, 1.0, -1.0])}


def pose_variants(P):
    """Every plausible reading of a stored 4x4: as-is or inverted, each with
    the camera axes flipped in the three det=+1 ways."""
    out = {}
    for tag, Q in (("c2w", P), ("w2c", np.linalg.inv(P))):
        for ftag, D in FLIPS.items():
            V = Q.copy()
            V[:3, :3] = Q[:3, :3] @ D
            out[tag + ftag] = V
    return out


def diagnose(poses, paths, backbone, device, n=10):
    from smr.backbones import get_backbone
    from smr.eval.trajectory import ate_rmse, auc_at, pairwise_pose_errors
    step = max(1, len(paths) // n)
    idx = list(range(0, len(paths), step))[:n]
    est = np.asarray(get_backbone(backbone, device=device)
                     .infer([paths[i] for i in idx]).poses, float)
    scores = {}
    for name in pose_variants(np.eye(4)):
        gt = np.stack([pose_variants(poses[i])[name] for i in idx])
        rra, rta = pairwise_pose_errors(est, gt)
        scores[name] = dict(auc30=auc_at(rra, rta), rra_med=float(np.median(rra)),
                            rta_med=float(np.median(rta)), ate=ate_rmse(est, gt))
    return idx, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["sevenscenes", "tum"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene", default="chess", help="7-Scenes scene name")
    ap.add_argument("--seq", default="1",
                    help="7-Scenes seq number, or a comma list (1,2,3) to "
                         "concatenate sessions of the same scene")
    ap.add_argument("--sequence", default="rgbd_dataset_freiburg1_desk",
                    help="TUM sequence directory name")
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--out", default="")
    ap.add_argument("--convention", default="c2w",
                    help="reading to store (see --diagnose-convention)")
    ap.add_argument("--diagnose-convention", action="store_true")
    ap.add_argument("--backbone", default="vggt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()

    if a.dataset == "sevenscenes":
        seqs = [int(x) for x in str(a.seq).split(",") if x]
        poses, paths, ids, dropped = load_sevenscenes(
            a.root, a.scene, seqs if len(seqs) > 1 else seqs[0])
        scene = a.scene if len(seqs) == 1 else f"{a.scene}_s" + "".join(f"{x:02d}" for x in seqs)
        seq = "+".join(f"seq-{x:02d}" for x in seqs)
    else:
        poses, paths, ids, dropped = load_tum(a.root, a.sequence, a.max_dt)
        scene, seq = a.sequence, ""
    c = poses[:, :3, 3]
    print(f"{a.dataset} {scene} {seq}: {len(poses)} frames with GT "
          f"({len(dropped)} dropped)")
    if dropped[:5]:
        print("  first drops:", dropped[:5])
    print(f"  extent (bbox diag) {np.linalg.norm(np.ptp(c, 0)):.2f} m, "
          f"path length {np.linalg.norm(np.diff(c, axis=0), axis=1).sum():.2f} m")
    from smr.eval.trajectory import find_revisit_pairs
    rp = find_revisit_pairs(poses, min_gap=60, max_dist=0.25, max_angle_deg=30)
    print(f"  revisit pairs (>=60 frames apart, <0.25 m, <30 deg): {len(rp)}")

    if a.diagnose_convention:
        print(f"  scoring conventions with {a.backbone} on {a.n} frames spread "
              f"over the sequence ...")
        idx, scores = diagnose(poses, paths, a.backbone, a.device, a.n)
        print(f"  {'convention':<10}{'AUC@30':>9}{'RRA med':>10}{'RTA med':>10}"
              f"{'ATE':>10}")
        for name, s in sorted(scores.items(), key=lambda kv: -kv[1]["auc30"]):
            print(f"  {name:<10}{s['auc30']:>9.1f}{s['rra_med']:>10.2f}"
                  f"{s['rta_med']:>10.2f}{s['ate']:>10.4f}")
        best = max(scores, key=lambda k: scores[k]["auc30"])
        print(f"  best: {best}  -> store with --convention {best}")
        if not a.out:
            return

    if not a.out:
        raise SystemExit("pass --out to write the npz")
    stored = np.stack([pose_variants(P)[a.convention] for P in poses])
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, poses=stored, image_paths=np.array(paths), frame_ids=ids,
             dataset=a.dataset, scene=scene, sequence=seq,
             convention=a.convention, n_dropped=len(dropped))
    print(f"  wrote {out} ({len(stored)} poses, convention {a.convention})")


if __name__ == "__main__":
    main()
