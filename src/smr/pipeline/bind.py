"""Algorithm 1 (Bind): anchor each view's pose on the scaffold, read h, and
write (h -> s, s -> h) with one-shot RLS; store surfels + state row."""
from __future__ import annotations

import numpy as np

from ..memory import BlockScaffold, RLSMemory, Novelty, Store, remap_offsets
from ..scene import surfels_from_view
from ..utils.geometry import R_to_euler_zyx


class BoundScene:
    def __init__(self, scaffold_state, block: BlockScaffold, mem: RLSMemory,
                 store: Store, novelty: Novelty, offsets,
                 novelty_floor=0.0, novelty_tau=0.2):
        self.ss, self.block, self.mem = scaffold_state, block, mem
        self.store, self.novelty, self.offsets = store, novelty, offsets
        self.novelty_floor, self.novelty_tau = novelty_floor, novelty_tau


def bind(backbone_out, scaffold_state, cam, scene_id="scene0",
         formation_extent=1.6, formation_spacing=0.4, stride=2,
         torus_N=32, N_h=1024, k=64, seed=0):
    ss = scaffold_state
    offsets = remap_offsets(scene_id, len(ss.modules)) * 0.0  # single scene: identity
    block = BlockScaffold([m.lam for m in ss.modules], torus_N=torus_N,
                          N_h=N_h, k=k, seed=seed)
    xs = block.form_grid(formation_extent, formation_spacing)
    nov = Novelty().fit(block)
    # The reconstruction residual of a VALID but off-lattice pose scales
    # with formation spacing; calibrate the detect gate against measured
    # lattice-midpoint probes so tau is spacing-robust.
    mids = (xs[:-1] + xs[1:]) / 2.0
    probes = [np.array([mids[len(mids) // 2], m, 0.0]) for m in mids[:4]]
    floor = max(nov.score(block.encode_pos(p)) for p in probes)
    tau = max(0.2, 3.0 * floor)
    mem = RLSMemory(N_h=N_h, N_s=448)
    store = Store()
    for i in range(backbone_out.poses.shape[0]):
        T = backbone_out.poses[i]
        ss.place_pose(T)
        xi = ss.state()
        h = block.h_of(block.encode_phases(ss.phases()))
        s = backbone_out.descriptor(i)
        mem.write(h, s)
        pts, cols = surfels_from_view(backbone_out.rgb[i], backbone_out.depth[i],
                                      backbone_out.mask[i], cam, stride=stride)
        store.add(xi, dict(idx=i, T=T, pts=pts, cols=cols, h=h, s=s))
    return BoundScene(ss, block, mem, store, nov, offsets,
                      novelty_floor=floor, novelty_tau=tau)
