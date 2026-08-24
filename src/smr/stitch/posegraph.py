"""Batch Sim(3) pose-graph optimisation over chunk nodes.

The classical back-end.  Each chunk is a node with a similarity S_c
(chunk frame -> world).  Each accepted measurement is an edge (c, k, Z)
with Z ~ inv(S_c) o S_k (maps chunk-k coordinates into chunk-c
coordinates), obtained by aligning the frames of chunk c that appeared in
chunk k's pass onto their poses in chunk c's own pass.  Sequential edges
come from the overlaps; loop edges from verified anchors.

Minimises  sum_edges rho( || vec( inv(Z) o inv(S_c) o S_k ) ||^2 )
with a Huber loss (f_scale 0.5 -- wide enough that a verified loop edge
with a 0.2 rad discrepancy is not down-weighted), S_0 fixed, initialised
from the streaming solution.
Same measurements as the streaming stitcher; the only difference is that
every node moves at once.  If this row is much better than the streaming
row, batch optimisation is buying something the amortised correction is
not; if the two are close, the streaming placement is doing the job.
"""
from __future__ import annotations

import numpy as np

from . import sim3


def _node_init(result, chunks):
    """Initial node similarities from a streaming result."""
    owner = np.asarray(result["owner"])
    frames = sorted(result["local"])
    est = result["est"]
    nodes = []
    for c in range(len(chunks)):
        mine = [i for i, g in enumerate(frames) if owner[i] == c]
        if len(mine) >= 2:
            A = np.stack([result["local"][frames[i]] for i in mine])
            B = est[mine]
            S, _ = sim3.fit_poses(A, B)
        else:
            S = sim3.identity()
        nodes.append(S)
    return nodes


def solve(result, chunks, huber=0.5, max_nfev=200, verbose=False,
          weight="equal"):
    """Returns (poses (N,4,4), info).  Requires scipy."""
    from scipy.optimize import least_squares

    edges = [e for e in result["edges"]]
    K = len(chunks)
    nodes0 = _node_init(result, chunks)
    if K < 2 or not edges:
        return result["est"].copy(), dict(n_edges=len(edges), cost0=0.0, cost=0.0)

    x0 = np.concatenate([sim3.to_vec(S) for S in nodes0[1:]])

    def unpack(x):
        S = [sim3.identity()]
        for c in range(1, K):
            S.append(sim3.from_vec(x[7 * (c - 1): 7 * c]))
        return S

    def residuals(x):
        S = unpack(x)
        r = []
        for e in edges:
            c, k, Z = e["c"], e["k"], e["Z"]
            E = sim3.compose(sim3.inverse(Z), sim3.compose(sim3.inverse(S[c]), S[k]))
            v = sim3.to_vec(E)
            # Every edge is a verified Sim(3) fit, so edges are weighted
            # equally by default.  Weighting by sqrt(#correspondences)
            # favours the sequential edges (8 overlap frames vs 4 anchor
            # frames) and, measured in simulation, leaves ~2x more loop
            # error than equal weighting -- the batch row must not be
            # handicapped against the streaming one.
            w = np.sqrt(max(1, e.get("n", 2))) if weight == "sqrt_n" else 1.0
            r.append(w * v)
        return np.concatenate(r)

    r0 = residuals(x0)
    sol = least_squares(residuals, x0, loss="huber", f_scale=huber,
                        max_nfev=max_nfev, verbose=1 if verbose else 0)
    S = unpack(sol.x)
    owner = result["owner"]
    frames = sorted(result["local"])
    poses = np.stack([sim3.apply(S[owner[i]], result["local"][g][None])[0]
                      for i, g in enumerate(frames)])
    return poses, dict(n_edges=len(edges), n_loop_edges=sum(e["kind"] == "loop" for e in edges),
                       cost0=float(0.5 * (r0 ** 2).sum()), cost=float(sol.cost),
                       nfev=int(sol.nfev), status=int(sol.status))
