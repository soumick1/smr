#!/usr/bin/env python3
"""Pilot A -- the experiment the paper turns on.

A long sequence is chunked into overlapping windows.  Each window is one
backbone forward pass, producing poses in ITS OWN reference frame and ITS
OWN scale gauge.  Turning those chunks into a single global trajectory is
the problem; we compare two ways of doing it:

  baseline  Umeyama/Sim(3) alignment of each chunk onto the PREVIOUS
            chunk's already-transformed poses, chained.  This is what
            everyone does, and its error compounds multiplicatively:
            T_global(k) = S_k o S_{k-1} o ... o S_1.

  smr       Each chunk is aligned onto poses read back OUT OF THE SCAFFOLD.
            Because the grid code is an absolute, periodic lattice, a
            decoded pose is snapped to that lattice: alignment error below
            half a cell is removed rather than carried forward.  The claim
            under test is that this BOUNDS drift accumulation instead of
            letting it compound.

The backbone runs ONCE per chunk and both methods consume the same cached
outputs, so the only variable between a backbone's two rows is the
stitching mechanism -- no re-inference, no stochastic difference.

    # real backbone (run one backbone per process: several of these repos
    # vendor forks of dust3r/croco and will shadow each other)
    python experiments/pilot_a.py --frames <dir> --backbone vggt \
        --chunk 16 --overlap 4 --gt <poses.npz>

    # harness validation with no GPU (see --simulate-chunks)
    python experiments/pilot_a.py --frames /tmp/dense40 --backbone synthetic \
        --simulate-chunks --chunk-noise 0.02
"""
import argparse, importlib.util, json, pathlib, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smr.dynamics import ScaffoldState                      # noqa: E402
from smr.eval.trajectory import (apply_sim3_to_poses,       # noqa: E402
                                 ate_rmse, auc_at, collinearity,
                                 drift_at_revisit, pairwise_pose_errors,
                                 sim3_from_poses, summarise)
from smr.utils.geometry import euler_zyx_to_R, make_T       # noqa: E402

PERIODS = [2.4, 3.2, 4.0]

_spec = importlib.util.spec_from_file_location(
    "t3", ROOT / "experiments" / "run_tier3_vggt.py")
t3 = importlib.util.module_from_spec(_spec)
_argv = sys.argv; sys.argv = ["t3"]; _spec.loader.exec_module(t3)
sys.argv = _argv


# ---------------------------------------------------------------- chunks --
def make_chunks(n, size, overlap):
    """Overlapping index windows covering [0, n)."""
    assert 0 <= overlap < size, "overlap must be smaller than the window"
    stride = size - overlap
    chunks, start = [], 0
    while start < n:
        idx = list(range(start, min(start + size, n)))
        if len(idx) < 3 and chunks:            # fold a runt into the last
            chunks[-1] = sorted(set(chunks[-1] + idx))
            break
        chunks.append(idx)
        if start + size >= n:
            break
        start += stride
    return chunks


def regauge(poses, depth=None):
    """Express poses in the chunk's own frame with its own scale gauge.

    Every backbone reports relative to its first view and normalises scale
    per call; a simulated chunk must do the same or the experiment is
    measuring nothing.
    """
    poses = np.asarray(poses, float).copy()
    T0inv = np.linalg.inv(poses[0])
    poses = np.einsum("ij,kjl->kil", T0inv, poses)
    scale = float(np.median(np.linalg.norm(poses[1:, :3, 3], axis=1))) \
        if len(poses) > 1 else 1.0
    if scale > 1e-9:
        poses[:, :3, 3] /= scale
    return poses


def perturb(poses, sigma, rng):
    """A small random similarity + per-pose jitter: stands in for backbone
    error when validating the harness without a GPU."""
    if sigma <= 0:
        return poses
    out = poses.copy()
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    ang = rng.normal(scale=sigma)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    out = apply_sim3_to_poses(out, 1.0 + rng.normal(scale=sigma),
                              R, rng.normal(scale=sigma, size=3))
    out[:, :3, 3] += rng.normal(scale=sigma * 0.25, size=out[:, :3, 3].shape)
    return out


def chunk_outputs(paths, chunks, backbone, device, simulate, noise, seed=0):
    """Per-chunk poses, each in its own frame and gauge, plus timings."""
    rng = np.random.default_rng(seed)
    outs, secs, peak = [], [], 0.0
    if simulate:
        full, T_gt, _ = t3.backbone_output(backbone, str(paths[0].parent),
                                           device)
        for idx in chunks:
            t0 = time.time()
            p = perturb(regauge(full.poses[idx]), noise, rng)
            outs.append(p)
            secs.append(time.time() - t0)
        return outs, secs, peak, (T_gt if T_gt is not None else full.poses)

    import torch
    from smr.backbones import get_backbone
    bb = get_backbone(backbone, device=device)
    for idx in chunks:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = bb.infer([str(paths[i]) for i in idx])
        secs.append(time.time() - t0)
        if torch.cuda.is_available():
            peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
        outs.append(np.asarray(out.poses, float))
    return outs, secs, peak, None


# --------------------------------------------------------------- stitching --
def stitch_baseline(chunk_poses, chunks):
    """Chain Sim(3) alignments onto the previous chunk's global poses."""
    glob = {}
    for k, (idx, poses) in enumerate(zip(chunks, chunk_poses)):
        if k == 0:
            for li, gi in enumerate(idx):
                glob[gi] = poses[li]
            continue
        ov = [(li, gi) for li, gi in enumerate(idx) if gi in glob]
        assert len(ov) >= 3, (
            f"chunk {k} shares only {len(ov)} frames with the trajectory so "
            f"far; Sim(3) needs 3 -- increase --overlap")
        A = poses[[li for li, _ in ov]]
        B = np.stack([glob[gi] for _, gi in ov])
        moved = apply_sim3_to_poses(poses, *sim3_from_poses(A, B))
        for li, gi in enumerate(idx):
            glob.setdefault(gi, moved[li])
    return np.stack([glob[i] for i in sorted(glob)])


def scaffold_decode(ss, near=None):
    """Read a pose back out of the scaffold's activity.

    Each grid module reports position only MODULO its period, so a decode
    that unwraps inside a single +-lambda/2 window is limited to a ~2-unit
    envelope -- far too small for a long trajectory, which then aliases and
    poisons the alignment.

    Continuity solves it, and is how grid codes are actually used for path
    integration: given the previously decoded position `near`, each module's
    phase is unwrapped to the branch closest to it, and the modules are
    averaged.  Range becomes unbounded provided consecutive frames move less
    than lambda_min/2 = 1.2 units -- comfortably true for video.  With
    near=None we fall back to the absolute (single-window) decode, which is
    correct for the first frame.
    """
    xi = ss.state()
    ph = xi[3:].reshape(3, 3)                    # [module][axis]
    if near is None:
        order = np.argsort(-np.asarray(PERIODS))
        x = np.zeros(3)
        for rank, m in enumerate(order):
            lam, frac = PERIODS[m], ph[m] / (2 * np.pi)
            if rank == 0:
                x = ((frac + 0.5) % 1.0 - 0.5) * lam
            else:
                x = x + lam * (((frac - x / lam) + 0.5) % 1.0 - 0.5)
    else:
        near = np.asarray(near, float)
        branches = []
        for m, lam in enumerate(PERIODS):
            raw = (ph[m] / (2 * np.pi)) * lam
            branches.append(near + ((raw - near + lam / 2) % lam) - lam / 2)
        x = np.mean(branches, axis=0)
    return make_T(euler_zyx_to_R(*xi[:3]), x)


def stitch_smr(chunk_poses, chunks, ring_N=256, torus_N=64):
    """Align each chunk onto poses decoded FROM THE SCAFFOLD.

    The values that seed the next alignment have been through the grid
    lattice, so per-chunk alignment error is snapped rather than carried --
    the mechanism whose effect this pilot is built to measure.
    """
    ss = ScaffoldState(periods=PERIODS, ring_N=ring_N, torus_N=torus_N,
                       seed=0, omega_max=0.16)
    ss.calibrate()
    glob, decoded = {}, {}

    last = {"x": None}

    def remember(gi, T):
        glob[gi] = T
        ss.place_pose(T)
        # unwrap against the last decoded position: the trajectory is
        # continuous, so this lifts the decode out of a single period
        decoded[gi] = scaffold_decode(ss, near=last["x"])
        last["x"] = decoded[gi][:3, 3]

    for k, (idx, poses) in enumerate(zip(chunks, chunk_poses)):
        if k == 0:
            for li, gi in enumerate(idx):
                remember(gi, poses[li])
            continue
        ov = [(li, gi) for li, gi in enumerate(idx) if gi in decoded]
        assert len(ov) >= 3, (
            f"chunk {k} shares only {len(ov)} frames with memory; "
            f"increase --overlap")
        A = poses[[li for li, _ in ov]]
        B = np.stack([decoded[gi] for _, gi in ov])          # from memory
        moved = apply_sim3_to_poses(poses, *sim3_from_poses(A, B))
        for li, gi in enumerate(idx):
            if gi not in glob:
                remember(gi, moved[li])
    return np.stack([glob[i] for i in sorted(glob)])


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="",
                    help="ignored when --gt carries image_paths")
    ap.add_argument("--backbone", default="vggt")
    ap.add_argument("--keyframe-stride", type=int, default=6,
                    help="evaluate on every Nth frame.  Consecutive video "
                         "frames are a few degrees apart, a near-degenerate "
                         "baseline these models cannot resolve; keyframes "
                         "give each chunk a wide, well-conditioned arc.  "
                         "Set 1 to reproduce the contiguous behaviour.")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--gt", default="", help="npz with a (K,4,4) c2w 'poses'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--simulate-chunks", action="store_true",
                    help="derive chunks from one pass + regauge (no GPU); "
                         "for validating the harness, never for a result")
    ap.add_argument("--chunk-noise", type=float, default=0.0,
                    help="simulated per-chunk backbone error (with "
                         "--simulate-chunks)")
    ap.add_argument("--json", default="outputs/reports/pilot_a.json")
    a = ap.parse_args()

    gt_npz = np.load(a.gt, allow_pickle=True) if a.gt else None
    if gt_npz is not None and "image_paths" in gt_npz:
        # take the frame list FROM the ground truth: pose i must belong to
        # image i, and a sorted directory listing does not guarantee that
        paths = [pathlib.Path(str(s)) for s in gt_npz["image_paths"]]
        d = paths[0].parent
    elif a.frames:
        d = pathlib.Path(a.frames)
        paths = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    else:
        raise SystemExit("pass --gt (preferred) or --frames")
    if a.max_frames:
        paths = paths[:a.max_frames]
    key = list(range(0, len(paths), max(1, a.keyframe_stride)))
    paths = [paths[i] for i in key]
    chunks = make_chunks(len(paths), a.chunk, a.overlap)
    simulated = a.simulate_chunks or a.backbone == "synthetic"
    if simulated:
        print("!" * 78)
        print("!! SIMULATED RUN -- NO MODEL IS EXECUTED AND NO IMAGE IS "
              "PERCEIVED.")
        if a.backbone == "synthetic":
            print("!! --backbone synthetic replays ground-truth poses from "
                  "gt.npz.")
        if a.simulate_chunks:
            print("!! --simulate-chunks slices those poses into windows and "
                  "adds noise.")
        print("!! These numbers validate the HARNESS ONLY. They are not "
              "results and")
        print("!! must never appear in a table. Real numbers need a real "
              "backbone and")
        print("!! real ground-truth poses on the GPU server.")
        print("!" * 78)
    print(f"scene {d.name} | {len(paths)} keyframes (every "
          f"{a.keyframe_stride} of {len(key) * a.keyframe_stride}) | "
          f"{len(chunks)} chunks of {a.chunk} (overlap {a.overlap}) | "
          f"backbone {a.backbone}")

    outs, secs, peak, gt_fallback = chunk_outputs(
        paths, chunks, a.backbone, a.device, a.simulate_chunks,
        a.chunk_noise)

    if gt_npz is not None:
        gt = np.asarray(gt_npz["poses"], float)[key]
    elif gt_fallback is not None:
        gt = np.asarray(gt_fallback, float)[key]
    else:
        raise SystemExit("no ground truth: pass --gt poses.npz")

    from smr.eval.trajectory import rotation_angle_deg
    steps = [rotation_angle_deg(gt[i, :3, :3].T @ gt[i + 1, :3, :3])
             for i in range(len(gt) - 1)]
    print(f"  median rotation between adjacent keyframes: "
          f"{np.median(steps):.1f} deg  (a few degrees is degenerate for "
          f"these models; ~10 or more is their operating regime)")
    col = (min(collinearity(gt[c[:a.overlap], :3, 3]) for c in chunks[1:])
           if len(chunks) > 1 else 1.0)
    print(f"  overlap spread {col:.3f} (0 = collinear); "
          f"alignment uses orientations, so this is diagnostic only")

    revisits = [(i, j) for i in range(len(gt)) for j in range(i + 20, len(gt))
                if np.linalg.norm(gt[i, :3, 3] - gt[j, :3, 3])
                < 0.15 * float(np.linalg.norm(np.ptp(gt[:, :3, 3], axis=0)))]

    # ---- per-chunk probe: is the BACKBONE any good before stitching? ----
    # RPE at delta=1 is dominated by pairs inside a chunk, so a large RPE
    # with a broken ATE cannot be blamed on the stitcher.  Score each chunk
    # against GT on its own, Sim(3)-aligned, so backbone quality and
    # stitching quality are never confused again.
    print(f"\n  per-chunk quality (backbone only, each aligned separately):")
    print(f"  {'chunk':>6}{'frames':>8}{'ATE':>10}{'AUC@30':>9}{'RRA med':>10}")
    probe = []
    for k, (idx, poses) in enumerate(zip(chunks, outs)):
        g = gt[idx]
        rra, rta = pairwise_pose_errors(poses, g)
        pr = dict(chunk=k, n=len(idx), ate=ate_rmse(poses, g),
                  auc30=auc_at(rra, rta), rra_med=float(np.median(rra)))
        probe.append(pr)
        print(f"  {k:>6}{len(idx):>8}{pr['ate']:>10.4f}{pr['auc30']:>9.1f}"
              f"{pr['rra_med']:>10.2f}")
    worst = max(probe, key=lambda r: r["ate"])
    if worst["auc30"] < 50:
        print(f"\n  *** The BACKBONE is failing inside chunks (worst chunk "
              f"AUC {worst['auc30']:.1f}). Stitching cannot repair that, and\n"
              f"  *** nothing below is a measurement of SMR. Change the chunk "
              f"geometry (--chunk / --keyframe-stride) until this is high.\n")

    rows = []
    for method, fn in (("backbone alone", stitch_baseline),
                       ("+ SMR", stitch_smr)):
        est = fn(outs, chunks)
        cols = summarise(est, gt)
        cols.update(backbone=a.backbone, method=method,
                    n_chunks=len(chunks),
                    drift_revisit=drift_at_revisit(est, gt, revisits[:50]),
                    peak_mem_gb=round(peak, 2),
                    sec_per_frame=round(sum(secs) / max(1, len(paths)), 3))
        rows.append(cols)

    hdr = (f"{'backbone':<11}{'method':<16}{'ATE':>9}{'RPE-t':>9}"
           f"{'RPE-r':>9}{'AUC@30':>8}{'revisit':>9}{'mem':>7}{'s/frm':>7}")
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['backbone']:<11}{r['method']:<16}{r['ate_rmse']:>9.4f}"
              f"{r['rpe_trans']:>9.4f}{r['rpe_rot_deg']:>9.3f}"
              f"{r['auc30']:>8.1f}{r['drift_revisit']:>9.4f}"
              f"{r['peak_mem_gb']:>7.2f}{r['sec_per_frame']:>7.3f}")
    print("=" * len(hdr))
    better = rows[1]["ate_rmse"] < rows[0]["ate_rmse"]
    print(f"ATE: +SMR is {'BETTER' if better else 'not better'} "
          f"({rows[1]['ate_rmse']:.4f} vs {rows[0]['ate_rmse']:.4f})")

    out = ROOT / a.json
    if simulated:                 # a simulated report can never sit in the
        out = out.with_name(out.stem + "_SIMULATED" + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)   # place a real one would
    out.write_text(json.dumps(dict(
        SIMULATED=simulated,
        WARNING=("harness validation only; no model executed"
                 if simulated else None),
        scene=str(d), frames=len(paths), chunk=a.chunk, overlap=a.overlap,
        backbone=a.backbone, chunk_noise=a.chunk_noise,
        n_revisit_pairs=len(revisits), rows=rows), indent=2))
    print(f"report -> {out}")
    if simulated:
        print("reminder: SIMULATED. Nothing above is a result.")


if __name__ == "__main__":
    main()
