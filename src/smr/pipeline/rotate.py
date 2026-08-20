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


def rotate(bound, T_query, cam: Camera, k_recall=2, record=False,
           settle_fiber=True):
    ss, block, store = bound.ss, bound.block, bound.store
    n_steps, trace = ss.drive_to(T_query, record=record)
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
                trace=trace, recalled=ids)
