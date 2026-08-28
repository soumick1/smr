"""Dense re-measurement of pose-graph edges from point clouds.

Every edge of the streaming stitcher is a Sim(3) fitted from a handful of
CAMERA POSES (8 shared frames for a junction, ~4 site frames for a loop).
This module re-measures each edge from thousands of pixel-exact 3D-3D
correspondences -- the same physical pixels unprojected in the two chunks'
own passes -- and returns a result whose edges carry the dense Z, ready
for the same robust batch solve.  Loop edges get one fresh mini-pass over
the spatially closest frames of the two chunks (chosen from the current
batch estimate), replacing the historical anchored pass.
"""
from __future__ import annotations

import numpy as np

from . import posegraph, sim3


def _unproject(rv, li, ys, xs):
    d = rv.depth[li][ys, xs]
    ok = d > 1e-6
    K = rv.intrinsics[li] if rv.intrinsics.ndim == 3 else rv.intrinsics
    X = (xs[ok] - K[0, 2]) / K[0, 0] * d[ok]
    Y = (ys[ok] - K[1, 2]) / K[1, 1] * d[ok]
    Pc = np.stack([X, Y, d[ok]], 1)
    return Pc @ rv.poses[li][:3, :3].T + rv.poses[li][:3, 3], ok


def _pair_frames(rv_a, la, rv_b, lb, ys, xs, cap=4000, rng=None):
    """Same pixels of the same frames unprojected in two passes."""
    A, B = [], []
    for i, j in zip(la, lb):
        da, db = rv_a.depth[i][ys, xs], rv_b.depth[j][ys, xs]
        m = (da > 1e-6) & (db > 1e-6)
        Ka = rv_a.intrinsics[i] if rv_a.intrinsics.ndim == 3 else rv_a.intrinsics
        Kb = rv_b.intrinsics[j] if rv_b.intrinsics.ndim == 3 else rv_b.intrinsics
        Pa = np.stack([(xs[m] - Ka[0, 2]) / Ka[0, 0] * da[m],
                       (ys[m] - Ka[1, 2]) / Ka[1, 1] * da[m], da[m]], 1)
        Pb = np.stack([(xs[m] - Kb[0, 2]) / Kb[0, 0] * db[m],
                       (ys[m] - Kb[1, 2]) / Kb[1, 1] * db[m], db[m]], 1)
        A.append(Pa @ rv_a.poses[i][:3, :3].T + rv_a.poses[i][:3, 3])
        B.append(Pb @ rv_b.poses[j][:3, :3].T + rv_b.poses[j][:3, 3])
    A, B = np.concatenate(A), np.concatenate(B)
    if len(A) > cap:
        sel = (rng or np.random.default_rng(0)).choice(len(A), cap, replace=False)
        A, B = A[sel], B[sel]
    return A, B


def remeasure(result, chunks, frame_paths, infer_fn, pixel_stride=6,
              loop_pass_frames=4, min_pts=200, verbose=False):
    """Returns a copy of `result` with every edge's Z re-fitted densely.

    frame_paths[g] is the image path of keyframe position g; infer_fn(paths)
    returns an object with .poses .depth .intrinsics.  Edges whose dense fit
    fails (too few valid pairs) keep their pose-fit Z.
    """
    rng = np.random.default_rng(0)
    passes = {}

    def own(ci):
        if ci not in passes:
            passes[ci] = infer_fn([frame_paths[g] for g in chunks[ci]])
        return passes[ci]

    # frame -> world from the current batch estimate, for loop frame choice
    # (solved lazily: only loop edges need it, and minimal results may lack
    # the keys the solver's initialiser reads)
    world_pos = None

    def get_world():
        nonlocal world_pos
        if world_pos is None:
            pg0, _ = posegraph.solve(result, chunks)
            world_pos = {g: pg0[g][:3, 3] for g in range(len(pg0))}
        return world_pos
    grid = None
    edges2 = []
    n_dense = 0
    for e in result["edges"]:
        c, k = int(e["c"]), int(e["k"])
        rv_c, rv_k = own(c), own(k)
        if grid is None:
            H, W = rv_c.depth.shape[1:3]
            ys, xs = np.mgrid[0:H:pixel_stride, 0:W:pixel_stride]
        e2 = dict(e)
        try:
            if e["kind"] == "seq":
                shared = [g for g in chunks[k] if g in set(chunks[c])]
                la = [chunks[k].index(g) for g in shared]
                lb = [chunks[c].index(g) for g in shared]
                A, B = _pair_frames(rv_k, la, rv_c, lb, ys, xs, rng=rng)
            else:
                wp = get_world()
                fc = sorted(chunks[c], key=lambda g: min(
                    np.linalg.norm(wp[g] - wp[h]) for h in chunks[k]))[:loop_pass_frames]
                fk = sorted(chunks[k], key=lambda g: min(
                    np.linalg.norm(wp[g] - wp[h]) for h in fc))[:loop_pass_frames]
                rv_m = infer_fn([frame_paths[g] for g in fc + fk])
                Ac, Bc = _pair_frames(rv_m, list(range(len(fc))), rv_c,
                                      [chunks[c].index(g) for g in fc], ys, xs, rng=rng)
                Ak, Bk = _pair_frames(rv_m, list(range(len(fc), len(fc) + len(fk))), rv_k,
                                      [chunks[k].index(g) for g in fk], ys, xs, rng=rng)
                if len(Ac) < min_pts or len(Ak) < min_pts:
                    edges2.append(e2); continue
                S_pc, _, _ = sim3.fit_points_robust(Ac, Bc)   # mini-pass -> chunk-c
                S_pk, _, _ = sim3.fit_points_robust(Ak, Bk)   # mini-pass -> chunk-k
                # Z: chunk-k -> chunk-c  =  (pass->c) o inv(pass->k)
                e2["Z"] = sim3.compose(S_pc, sim3.inverse(S_pk))
                e2["w_scale"] = 1.0
                edges2.append(e2); n_dense += 1
                continue
            if len(A) < min_pts:
                edges2.append(e2); continue
            S, keep, info = sim3.fit_points_robust(A, B)
            e2["Z"] = S            # A (chunk-k coords) -> B (chunk-c coords)
            e2["w_scale"] = 1.0
            n_dense += 1
        except Exception:
            pass
        edges2.append(e2)
    out = dict(result); out["edges"] = edges2
    if verbose:
        print(f"    dense edges: {n_dense}/{len(edges2)} re-measured "
              f"({len(passes)} chunk passes)")
    return out
