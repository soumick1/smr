#!/usr/bin/env python3
r"""DTU point-cloud evaluation (official protocol, offline).

Protocol (Aanaes et al.; the MATLAB code the published numbers come from):
prediction thinned to 0.2 mm voxels; Accuracy = mean distance from prediction
points inside the bounding box AND the observability mask to the scan, with
distances >= MaxDist (20 mm) discarded; Completeness = mean distance from scan
points above the ground plane to the prediction, same cut-off; Overall = mean.
Calibrated against the furu/tola/camp reference clouds of the SampleSet.

Gauge (ours; the clouds arrive from a camera fit): Sim(3) refinement onto the
scan, evaluated as candidates -- camera alignment as-is, region-restricted
point-to-plane ICP, and the same from a robust coarse initialisation -- kept
by fit quality (median NN distance of in-region points).  A refinement that
changes scale by more than --icp-max-scale or moves the cloud more than
--icp-max-move mm is rejected.  Order never decides; the camera alignment is
kept when nothing improves the fit.

    python scripts/dtu_eval.py --pred a.ply [b.ply ...] [--tags A B] --scan 1 \
        --sampleset ~/data/dtu/SampleSet/MVS\ Data --points-dir ~/data/dtu/Points/stl --icp 50
With --tags, each result is preceded by "== <tag> scan<N> ==" so several clouds of
one scan are scored in one process (GT, normals and trees loaded once).
"""
import argparse, pathlib, re, sys, time
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
              "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def read_ply(p):
    """xyz of a PLY file: ASCII or binary little-endian, with or without plyfile."""
    raw = pathlib.Path(p).read_bytes()
    if raw[:3] != b"ply":
        raise ValueError(f"{p}: not a PLY file")
    try:
        import plyfile
        v = plyfile.PlyData.read(str(p))["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], -1).astype(float)
    except ImportError:
        pass
    head, body = raw.split(b"end_header\n", 1)
    hdr = head.decode("ascii", "replace")
    n = int(re.search(r"element vertex (\d+)", hdr).group(1))
    if "format ascii" in hdr:
        return np.loadtxt(body.splitlines()[:n], usecols=(0, 1, 2))
    vert = hdr.split("element vertex")[1].split("element")[0]
    props = [(nm, _PLY_TYPES[t]) for t, nm in re.findall(r"property (\w+) (\w+)", vert)]
    arr = np.frombuffer(body, dtype=np.dtype(props), count=n)
    return np.stack([arr["x"], arr["y"], arr["z"]], -1).astype(float)


def thin(P, voxel):
    """First point per voxel (the official 0.2 mm 'reduce'), via a collision-free hashed key."""
    if voxel <= 0 or len(P) == 0:
        return P
    g = np.floor(P / voxel).astype(np.int64)
    g -= g.min(0)
    if (g.max(0) >= 2 ** 21).any():
        _, keep = np.unique(g, axis=0, return_index=True)          # fallback for absurd extents
    else:
        key = (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]
        _, keep = np.unique(key, return_index=True)
    return P[np.sort(keep)]


class GT:
    """Scan, masks, plane, and the reusable pieces of the gauge for one DTU scan."""

    def __init__(self, scan, sampleset, points_dir, icp_mode):
        ss = pathlib.Path(sampleset)
        stl = pathlib.Path(points_dir) if points_dir else ss / "Points" / "stl"
        self.scan = scan
        self.gt = read_ply(stl / f"stl{scan:03d}_total.ply")
        M = loadmat(ss / "ObsMask" / f"ObsMask{scan}_10.mat")
        self.mask = np.asarray(M["ObsMask"])
        self.BB = np.asarray(M["BB"], float).reshape(2, 3)
        self.Res = float(np.asarray(M["Res"]).squeeze())
        self.pl = np.asarray(loadmat(ss / "ObsMask" / f"Plane{scan}.mat")["P"], float).reshape(4)
        self.tree_full = cKDTree(self.gt)                                # Accuracy queries
        rng = np.random.RandomState(0)
        self.Gs = self.gt[rng.choice(len(self.gt), min(len(self.gt), 200_000), replace=False)]
        self.tree = cKDTree(self.Gs)                                     # gauge queries
        self.N = None
        if icp_mode == "plane":                                          # GT normals by local PCA (k=12)
            _, nn = self.tree.query(self.Gs, k=12, workers=-1)
            Q = self.Gs[nn] - self.Gs[nn].mean(1, keepdims=True)
            self.N = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))[1][:, :, 0]
        self.above = self.gt @ self.pl[:3] + self.pl[3] > 0
        ga = self.gt[self.above]
        self.gt_sub_above = ga[np.random.RandomState(2).choice(len(ga), min(len(ga), 50_000), replace=False)]

    def in_region(self, P):
        """Official evaluation region: inside the BB and inside the ObsMask grid."""
        g = np.floor((P - self.BB[0:1]) / self.Res).astype(int)
        ok = ((P >= self.BB[0:1]) & (P < self.BB[1:2])).all(1)
        gi = np.clip(g, 0, np.array(self.mask.shape) - 1)
        return ok & self.mask[gi[:, 0], gi[:, 1], gi[:, 2]].astype(bool)

    def fit_quality(self, P, max_dist=20.0):
        """Figure of merit of an alignment: the truncated Chamfer mean on subsamples -- the mean NN
        distance of in-region prediction points to the scan plus the mean NN distance of scan
        points (above the plane) to the prediction, both with the protocol's MaxDist cut.  This is
        the objective the gauge should minimise; medians (v134-v136) let a shrunk or slid cloud
        that overlaps the densest part of the object win with a wrong shape."""
        reg = self.in_region(P)
        if reg.sum() < 2000:
            return np.inf
        rs = np.random.RandomState(1)
        sub = P[reg][rs.choice(reg.sum(), min(reg.sum(), 50_000), replace=False)]
        d_pg = self.tree.query(sub, workers=-1)[0]
        d_gp = cKDTree(P[reg]).query(self.gt_sub_above, workers=-1)[0]
        return float(0.5 * (np.minimum(d_pg, max_dist).mean() + np.minimum(d_gp, max_dist).mean()))


def run_icp(pred0, coarse, G, a):
    """Region-restricted Sim(3) refinement; returns (aligned, info) or (None, reason)."""
    rng = np.random.RandomState(0)
    pred = pred0.copy()
    init_shift = np.zeros(3)
    init_scale = 1.0
    if coarse:
        # Robust similarity initialisation that makes no assumption about where the cloud
        # arrived: (1) match the median of ALL prediction points to the scan's median
        # (translation only -- the table would bias a scale from all points); (2) with the
        # support shrunk to prediction points within 60 mm, then 40 mm, of the scan, match
        # medians and the median radial extents (a robust scale).  Composition tracked so
        # the total similarity is reported and capped like any other refinement.
        mg = np.median(G.gt_sub_above, 0); eg = np.median(np.linalg.norm(G.gt_sub_above - mg, axis=1))
        S, Tt = 1.0, np.zeros(3)
        sub0 = pred[rng.choice(len(pred), min(len(pred), 100_000), replace=False)]
        cur = sub0.copy()
        for cap in (None, 60.0, 40.0):
            if cap is None:
                sel = np.ones(len(cur), bool); s_step = 1.0
            else:
                sel = G.tree.query(cur, workers=-1)[0] < cap
                if sel.sum() < 2000:
                    break
                ep = np.median(np.linalg.norm(cur[sel] - np.median(cur[sel], 0), axis=1))
                s_step = float(np.clip(eg / max(ep, 1e-9), 0.25, 4.0))
            mp = np.median(cur[sel], 0)
            cur = mg + s_step * (cur - mp)                     # new = s*old + (mg - s*mp)
            S *= s_step; Tt = s_step * Tt + (mg - s_step * mp)
        pred = S * pred + Tt
        init_scale, init_shift = S, Tt
    region = G.in_region(pred)
    cand = pred[region] if region.sum() >= 5000 else pred
    P = cand[rng.choice(len(cand), min(len(cand), 60_000), replace=False)].copy()
    T_s, T_R, T_t = 1.0, np.eye(3), np.zeros(3)
    coarse_iters = 5 if coarse else 0
    it = -1
    for it in range(a.icp):
        d_nn, idx = G.tree.query(P, workers=-1)
        cap = a.max_dist * (3.0 if it < coarse_iters else 1.0)
        near = d_nn < cap
        if near.sum() < 1000:
            break
        keep_nn = near & (d_nn < np.percentile(d_nn[near], 80))
        A, B = P[keep_nn], G.Gs[idx[keep_nn]]
        if a.icp_mode == "plane" and it >= coarse_iters:
            Nn = G.N[idx[keep_nn]]
            J = np.concatenate([np.cross(A, Nn), Nn, (A * Nn).sum(1, keepdims=True)], 1)
            b = -((A - B) * Nn).sum(1)
            x = np.linalg.lstsq(J, b, rcond=None)[0]
            th, tvec, ds = x[:3], x[3:6], float(np.clip(x[6], -0.05, 0.05))
            ang = np.linalg.norm(th)
            K = np.array([[0, -th[2], th[1]], [th[2], 0, -th[0]], [-th[1], th[0], 0]])
            R = np.eye(3) + (np.sin(ang) / ang) * K + ((1 - np.cos(ang)) / ang ** 2) * (K @ K) if ang > 1e-12 else np.eye(3)
            s = 1.0 + ds
        else:
            muA, muB = A.mean(0), B.mean(0)
            A0, B0 = A - muA, B - muB
            U, S, Vt = np.linalg.svd(A0.T @ B0)
            D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ D @ U.T
            s = float(np.clip((S * np.diag(D)).sum() / (A0 ** 2).sum(), 0.95, 1.05))
            tvec = muB - s * (R @ muA)
        P_new = (s * (R @ P.T)).T + tvec
        step = np.linalg.norm(P_new - P, axis=1).mean()
        P = P_new
        T_R = R @ T_R; T_s = s * T_s; T_t = s * (R @ T_t) + tvec
        if step < a.icp_tol:
            break
    moved = np.linalg.norm((T_s * (T_R @ cand[:2000].T)).T + T_t - cand[:2000], axis=1).mean()
    if abs(np.log(T_s * init_scale)) > a.icp_max_scale or moved > a.icp_max_move:
        return None, f"scale {T_s * init_scale:.4f}, mean move {moved:.1f} mm"
    out = (T_s * (T_R @ pred.T)).T + T_t
    info = (f"icp({it + 1}/{a.icp},{a.icp_mode}{',coarse' if coarse else ''}): scale {T_s * init_scale:.5f}  "
            f"|t| {np.linalg.norm(T_t + init_shift):.3f} mm  (mean move {moved:.2f} mm, init scale "
            f"{init_scale:.3f} shift {np.linalg.norm(init_shift):.1f} mm, {region.mean() * 100:.0f}% of pred in region)")
    return out, info


def refine(pred, G, a, log):
    """Best-of-candidates gauge: camera alignment, plain refinement, coarse-initialised refinement."""
    cands = [("camera alignment", pred, G.fit_quality(pred))]
    for coarse in ([False, True] if a.icp_init == "auto" else [False]):
        res, info = run_icp(pred, coarse, G, a)
        if res is None:
            log(f"icp({a.icp}{',coarse' if coarse else ''}): REJECTED ({info})")
        else:
            cands.append((info, res, G.fit_quality(res)))
    best = min(cands, key=lambda c: c[2])
    if best[0] == "camera alignment":
        others = [c[2] for c in cands[1:]]
        log(f"icp: no refinement improved the fit (best candidate {min(others) if others else float('nan'):.2f} mm vs "
            f"camera {cands[0][2]:.2f} mm); keeping camera alignment")
        return pred
    log(best[0] + f"  [fit {best[2]:.2f} mm vs camera {cands[0][2]:.2f} mm]")
    return best[1]


def evaluate(pred, G, a):
    ok = G.in_region(pred)
    dp = G.tree_full.query(pred[ok], workers=-1)[0]
    acc = dp[dp < a.max_dist].mean()
    dg = cKDTree(pred).query(G.gt[G.above], workers=-1)[0]
    comp = dg[dg < a.max_dist].mean()
    return acc, comp, ok.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="one or more point clouds of the same scan")
    ap.add_argument("--tags", nargs="*", default=None, help="one tag per --pred; prints '== tag scanN ==' before each result")
    ap.add_argument("--scan", type=int, required=True)
    ap.add_argument("--sampleset", required=True); ap.add_argument("--max-dist", type=float, default=20.0)
    ap.add_argument("--points-dir", default=None, help="dir with stlXXX_total.ply (Points.zip)")
    ap.add_argument("--down", type=float, default=0.2, help="pred downsample voxel (mm); 0=off")
    ap.add_argument("--icp", type=int, default=0, help="Sim(3) ICP refinement: max iterations (0=off)")
    ap.add_argument("--icp-mode", choices=["plane", "point"], default="plane")
    ap.add_argument("--icp-tol", type=float, default=0.005, help="stop when the mean per-iteration move is below this (mm)")
    ap.add_argument("--icp-init", choices=["auto", "none"], default="auto",
                    help="auto: also try a coarse centroid initialisation; the candidate with the best fit wins")
    ap.add_argument("--icp-max-scale", type=float, default=0.5,
                    help="sanity cap on |log scale| of a refinement (fit quality decides; v134's 0.05 refused legitimate 10-25%% corrections)")
    ap.add_argument("--icp-max-move", type=float, default=300.0, help="sanity cap on the mean move (mm)")
    a = ap.parse_args()
    if a.tags is not None and len(a.tags) != len(a.pred):
        raise SystemExit("--tags must have one entry per --pred")
    t0 = time.time()
    G = GT(a.scan, a.sampleset, a.points_dir, a.icp_mode)
    for i, p in enumerate(a.pred):
        if a.tags:
            print(f"== {a.tags[i]} scan{a.scan} ==")
        if not pathlib.Path(p).exists():
            print(f"scan{a.scan}: MISSING {p}"); continue
        pred = thin(read_ply(p), a.down)
        if a.icp > 0:
            pred = refine(pred, G, a, print)
        acc, comp, kept = evaluate(pred, G, a)
        print(f"scan{a.scan}: Acc {acc:.3f}  Comp {comp:.3f}  Overall {(acc + comp) / 2:.3f}  "
              f"(pred kept {kept * 100:.1f}%, MaxDist {a.max_dist})")
    print(f"# {len(a.pred)} cloud(s) in {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
