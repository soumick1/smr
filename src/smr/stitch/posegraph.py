"""Sim(3) pose-graph optimisation over chunk nodes -- batch and local.

Each chunk is a node with a similarity S_c (chunk frame -> world).  Each
accepted measurement is an edge (c, k, Z) with Z ~ inv(S_c) o S_k (maps
chunk-k coordinates into chunk-c coordinates), obtained by aligning the
frames of chunk c that appeared in chunk k's pass onto their poses in
chunk c's own pass.  Sequential edges come from the overlaps; loop edges
from anchors that passed the consistency gate.

Minimises  sum_edges rho( || W vec( inv(Z) o inv(S_c) o S_k ) ||^2 )
with a Huber loss (f_scale 0.5), where W zeroes the log-scale component
of an edge whose anchors could not measure scale (w_scale = 0).  Edges
are otherwise weighted equally: every edge is a verified Sim(3) fit, and
weighting by correspondence count favoured the 8-frame overlaps over the
4-frame loops and left ~2x more loop error in simulation.

`solve_nodes` optimises a chosen set of FREE nodes with the rest fixed;
the streaming stitcher calls it on the chunks since the last closure
(local relaxation, cost O(stretch)), and `solve` calls it on every node
but the first (the batch upper bound).
"""
from __future__ import annotations

import numpy as np

from . import sim3


def solve_nodes(nodes, edges, free, huber=0.5, max_nfev=200):
    """nodes: {c: Sim3} initial values; edges: dicts with c, k, Z, w_scale;
    free: node ids to optimise (others fixed).  Returns {c: Sim3}."""
    from scipy.optimize import least_squares
    free = [c for c in free if c in nodes]
    if not free or not edges:
        return dict(nodes)
    x0 = np.concatenate([sim3.to_vec(nodes[c]) for c in free])
    fixed = {c: S for c, S in nodes.items() if c not in free}

    def unpack(x):
        S = dict(fixed)
        for i, c in enumerate(free):
            S[c] = sim3.from_vec(x[7 * i: 7 * i + 7])
        return S

    def residuals(x):
        S = unpack(x)
        r = []
        for e in edges:
            if e["c"] not in S or e["k"] not in S:
                continue
            E = sim3.compose(sim3.inverse(e["Z"]),
                             sim3.compose(sim3.inverse(S[e["c"]]), S[e["k"]]))
            v = sim3.to_vec(E)
            v[0] *= e.get("w_scale", 1.0)
            r.append(v)
        return np.concatenate(r) if r else np.zeros(1)

    sol = least_squares(residuals, x0, loss="huber", f_scale=huber,
                        max_nfev=max_nfev)
    return unpack(sol.x)


def _node_init(result, chunks):
    owner = np.asarray(result["owner"])
    frames = sorted(result["local"])
    est = result["est"]
    nodes = {}
    for c in range(len(chunks)):
        mine = [i for i, g in enumerate(frames) if owner[i] == c]
        if len(mine) >= 2:
            A = np.stack([result["local"][frames[i]] for i in mine])
            S, _ = sim3.fit_poses(A, est[mine])
        else:
            S = sim3.identity()
        nodes[c] = S
    return nodes


def solve(result, chunks, huber=0.5, max_nfev=200, verbose=False,
          robust_reject=True, robust_abs=0.3):
    """Batch solve over every node but the first, then drop outlier loop
    edges and solve again (robust_reject).  Returns (poses, info)."""
    edges = list(result["edges"])
    K = len(chunks)
    nodes0 = _node_init(result, chunks)
    if K < 2 or not edges:
        return result["est"].copy(), dict(n_edges=len(edges), n_loop_edges=0,
                                           cost0=0.0, cost=0.0)

    def cost(nodes):
        tot = 0.0
        for e in edges:
            E = sim3.compose(sim3.inverse(e["Z"]),
                             sim3.compose(sim3.inverse(nodes[e["c"]]), nodes[e["k"]]))
            v = sim3.to_vec(E)
            v[0] *= e.get("w_scale", 1.0)
            tot += 0.5 * float(v @ v)
        return tot

    nodes = solve_nodes(nodes0, edges, list(range(1, K)), huber=huber,
                        max_nfev=max_nfev)
    # ROBUST PASS: a single wrong loop closure dominated the aliasing runs
    # (TUM floor / room).  After the first solve, loop edges whose residual
    # is far above the rest (> max(3 x median, robust_abs)) are dropped and
    # the graph is solved again -- the batch counterpart of a closure the
    # stream should not have accepted.
    dropped = []
    if robust_reject:
        def res_norm(e, nd):
            E = sim3.compose(sim3.inverse(e["Z"]), sim3.compose(sim3.inverse(nd[e["c"]]), nd[e["k"]]))
            v = sim3.to_vec(E); v[0] *= e.get("w_scale", 1.0)
            return float(np.linalg.norm(v))
        rn = np.array([res_norm(e, nodes) for e in edges])
        med = float(np.median(rn)) + 1e-9
        keep = [e for e, r_ in zip(edges, rn)
                if not (e["kind"] == "loop" and r_ > max(3.0 * med, robust_abs))]
        dropped = [e for e in edges if e not in keep]
        if dropped:
            edges = keep
            nodes = solve_nodes(nodes0, edges, list(range(1, K)), huber=huber,
                                max_nfev=max_nfev)
    owner = result["owner"]
    frames = sorted(result["local"])
    poses = np.stack([sim3.apply(nodes[owner[i]], result["local"][g][None])[0]
                      for i, g in enumerate(frames)])
    return poses, dict(n_edges=len(edges),
                       n_loop_edges=sum(e["kind"] == "loop" for e in edges),
                       n_dropped=len(dropped),
                       cost0=cost(nodes0), cost=cost(nodes))
