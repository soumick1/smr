"""Algorithm 2 (Rotate): drive the bumps to the query pose, settle the fiber
through the scaffold loop (rings untouched -- Cor. base-blind), recall the
nearest bound views, rigidly transform their surfels, and splat."""
from __future__ import annotations

import numpy as np

from ..scene import Camera, splat, transform
from ..utils.geometry import euler_zyx_to_R, make_T


def xi_to_pose(xi, periods, decode_pos_fn):
    R = euler_zyx_to_R(*xi[:3])
    x = decode_pos_fn(xi)
    return make_T(R, x)


def _ls_position(ss):
    """Least-squares coherent projection of the module phases.

    The coherent basis is per-axis block structured, so for any phase
    perturbation orthogonal to C the per-axis weighted residual against the
    weights w_m = 2*pi/lam_m is exactly zero -- the LS fit nulls incoherent
    noise, unlike the chained unwrap (which leaks it into the refinement
    steps).  Chained decode supplies the unwrap point; one Gauss-Newton
    step per axis is then closed form."""
    import numpy as np
    x = ss.decode_position()
    ph = (np.stack([m.phases() for m in ss.modules]) - ss.offsets)
    lam = np.array([m.lam for m in ss.modules])
    w = 2 * np.pi / lam
    for _ in range(3):
        r = (ph - w[:, None] * x[None, :] + np.pi) % (2 * np.pi) - np.pi
        x = x + (w @ r) / (w @ w)
    return x


def coherent_reanchor(bound, tau=None, steps=80):
    """Detect -> correct: the loop's reconstruction residual (Novelty) flags
    incoherent fiber error; correction is re-anchoring every module on the
    single position best explained by the current phases (the chained
    residue decode IS the coherent projection).  Coherent errors are, by
    Thm. selective, invisible to the detector and are NOT touched here --
    only landmark re-observation can fix those (Cor. base-blind makes the
    same true of orientation).  Returns a small diagnostic dict."""
    import numpy as np
    ss, block = bound.ss, bound.block
    if tau is None:
        tau = getattr(bound, "novelty_tau", 0.2)
    score = bound.novelty.score(block.encode_phases(ss.phases()))
    if score <= tau:
        return dict(corrected=False, score_before=score, score_after=score)
    x_hat = _ls_position(ss)
    for m, mod in enumerate(ss.modules):
        mod.place(x_hat, offset=ss.offsets[m], steps=steps)
    score_after = bound.novelty.score(block.encode_phases(ss.phases()))
    return dict(corrected=True, score_before=score, score_after=score_after)


def rotate(bound, T_query, cam: Camera, k_recall=2, record=False,
           settle_fiber=False, correct=True, novelty_tau=None):
    ss, block, store = bound.ss, bound.block, bound.store
    n_steps, trace = ss.drive_to(T_query, record=record)
    diag = coherent_reanchor(bound, tau=novelty_tau) if correct else \
        dict(corrected=False)
    if settle_fiber:
        g = block.settle(block.encode_phases(ss.phases()))
        ph = block.decode_phases(g)
        for m, mod in enumerate(ss.modules):        # re-anchor fiber ONLY:
            mod.torus.place((ph[m, 0], ph[m, 1]), steps=60)
            mod.zring.place(ph[m, 2], steps=60)     # rings untouched
        xi = np.concatenate([ss.decode_euler(), ph.ravel()])
    else:
        xi = ss.state()
    ids = store.nearest(xi, k=k_recall)
    P, C = [], []
    for j in ids:
        pay = store.content[j]
        P.append(transform(pay["pts"], pay["T"], T_query))
        C.append(pay["cols"])
    Pw, Cw = np.concatenate(P), np.concatenate(C)
    rgb, depth, mask = splat(Pw, Cw, T_query, cam)
    return dict(rgb=rgb, depth=depth, mask=mask, xi=xi, n_steps=n_steps,
                trace=trace, recalled=ids, correction=diag)
