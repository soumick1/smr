"""The SMR memory used as a place index for chunk stitching.

Two implementations behind one interface, so the paper can ablate them:

  ScaffoldIndex    the SMR memory.  A view's pose is turned into a grid-code
                   ADDRESS h (fixed random projection of the module bumps,
                   k-winners-take-all -- Vector-HaSH), its descriptor s is
                   bound to that address by one-shot RLS in both directions,
                   and a new view proposes anchors by cueing s -> h and
                   reading which stored addresses it lands on.  Views taken
                   at the same PLACE share addresses, so a revisit proposes
                   the earlier views of that place even when the frames are
                   not identical.

  DescriptorIndex  the control: the same descriptors, plain cosine nearest
                   neighbour, no address.  If it stitches equally well the
                   address machinery is not what buys the trajectory result,
                   and the paper says so.

Both expose add / propose / near_pose / update_pose / pose.
"""
from __future__ import annotations

import numpy as np

from ..memory import BlockScaffold, RLSMemory
from ..utils.geometry import R_to_euler_zyx

PERIODS = (2.4, 3.2, 4.0)


class DescriptorIndex:
    name = "descriptor"

    def __init__(self, **kw):
        self.ids, self.T, self.S = [], {}, []
        self._adj_cos = []

    def __len__(self):
        return len(self.ids)

    def add(self, gi, T, s):
        if self.ids:
            self._adj_cos.append(float(self.S[-1] @ s))
        self.ids.append(int(gi))
        self.T[int(gi)] = np.asarray(T, float).copy()
        self.S.append(np.asarray(s, float))

    def _scores(self, s):
        return np.stack(self.S) @ s

    def propose(self, s, exclude=(), top=5, desc_thresh=None):
        if not self.ids:
            return []
        sc = self._scores(s)
        cos = np.stack(self.S) @ s
        thr = self.desc_thresh(desc_thresh)
        order = np.argsort(-sc)
        out = []
        for r in order:
            gi = self.ids[r]
            if gi in exclude or cos[r] < thr:
                continue
            out.append((gi, float(sc[r]), float(cos[r])))
            if len(out) >= top:
                break
        return out

    def desc_thresh(self, given=None):
        """Descriptor-cosine gate for a proposal: an absolute value if given,
        else the 5th percentile of the cosines between CONSECUTIVELY bound
        views (measured on this sequence at bind time), floored at 0.3.  A
        cheap pre-filter only -- geometry does the real verification."""
        if given is not None:
            return float(given)
        if len(self._adj_cos) < 5:
            return 0.3
        return max(0.3, float(np.percentile(self._adj_cos, 5)))

    def near_pose(self, T, k=4, exclude=()):
        """Stored views whose centre is nearest the pose's centre."""
        ids = [g for g in self.ids if g not in exclude]
        if not ids:
            return []
        c = np.stack([self.T[g][:3, 3] for g in ids])
        d = np.linalg.norm(c - np.asarray(T, float)[:3, 3], axis=1)
        return [ids[i] for i in np.argsort(d)[:k]]

    def update_pose(self, gi, T_new):
        self.T[int(gi)] = np.asarray(T_new, float).copy()

    def pose(self, gi):
        return self.T[int(gi)]


class ScaffoldIndex(DescriptorIndex):
    name = "scaffold"

    def __init__(self, periods=PERIODS, torus_N=32, N_h=2048, k=None,
                 seed=0, encode="template", scaffold_state=None,
                 rls_lam=1e2, desc_dim=448):
        super().__init__()
        self.block = BlockScaffold(list(periods), torus_N=torus_N, N_h=N_h,
                                   k=(k or max(8, N_h // 16)), seed=seed)
        self.mem = RLSMemory(N_h=N_h, N_s=int(desc_dim), lam=rls_lam)
        self.encode = encode
        self.ss = scaffold_state          # only for encode="dynamics"
        self.H = []                       # stored addresses (unit rows)
        self.XI = []                      # stored states [euler, phases]

    # ------------------------------------------------------------ encoding
    def address(self, T):
        """Grid-code address of a pose.

        encode="template": analytic bump templates at the phases of the
        pose's position (BlockScaffold.encode_pos) -- the address the
        continuous-attractor dynamics settle to, without running them.
        encode="dynamics": place the real bumps (ScaffoldState.place_pose)
        and read their phases; identical up to the decode floor, ~100x
        slower.  The pilot uses the template; the equivalence is tested.
        """
        T = np.asarray(T, float)
        if self.encode == "dynamics":
            assert self.ss is not None, "encode='dynamics' needs a ScaffoldState"
            self.ss.place_pose(T)
            ph = self.ss.phases()
        else:
            ph = self.block.phases_of_pos(T[:3, 3])
        g = self.block.encode_phases(ph)
        h = self.block.h_of(g)
        xi = np.concatenate([np.array(R_to_euler_zyx(T[:3, :3])), ph.ravel()])
        return h, xi

    def add(self, gi, T, s):
        super().add(gi, T, s)
        h, xi = self.address(T)
        self.mem.write(h, np.asarray(s, float))
        self.H.append(h)
        self.XI.append(xi)

    def _scores(self, s):
        """Cue s -> address through the RLS map, then score every stored
        address by overlap with the cue: the propose stage of the
        propose-verify relocalisation."""
        cue = self.mem.cue_h(np.asarray(s, float))
        cue = cue / (np.linalg.norm(cue) + 1e-12)
        return np.stack(self.H) @ cue

    def update_pose(self, gi, T_new):
        """Re-anchor a stored view after a loop-closure correction: the pose
        annotation and the state row move; the appearance<->address
        association written at first sight is kept (the address is the
        view's label in memory, and the RLS map still cues it)."""
        super().update_pose(gi, T_new)
        r = self.ids.index(int(gi))
        _, xi = self.address(T_new)
        self.XI[r] = xi


def make_index(kind, **kw):
    if kind in ("scaffold", "smr", "address"):
        return ScaffoldIndex(**kw)
    if kind in ("descriptor", "plain", "nn"):
        return DescriptorIndex()
    raise KeyError(kind)


def scaffold_decode(ss, periods=PERIODS, near=None):
    """Read a pose back out of the scaffold's activity (continuity decode).

    Each grid module reports position only MODULO its period, so a decode
    that unwraps inside a single +-lambda/2 window is limited to a ~2-unit
    envelope -- far too small for a long trajectory, which then aliases.
    Given the previously decoded position `near`, each module's phase is
    unwrapped to the branch closest to it and the modules are averaged;
    range becomes unbounded provided consecutive frames move less than
    lambda_min/2 = 1.2 units.  With near=None the absolute (single-window)
    decode is used, correct for the first frame.
    """
    from ..utils.geometry import euler_zyx_to_R, make_T
    xi = ss.state()
    ph = xi[3:].reshape(len(periods), 3)
    periods = list(periods)
    if near is None:
        order = np.argsort(-np.asarray(periods))
        x = np.zeros(3)
        for rank, m in enumerate(order):
            lam, frac = periods[m], ph[m] / (2 * np.pi)
            if rank == 0:
                x = ((frac + 0.5) % 1.0 - 0.5) * lam
            else:
                x = x + lam * (((frac - x / lam) + 0.5) % 1.0 - 0.5)
    else:
        near = np.asarray(near, float)
        branches = []
        for m, lam in enumerate(periods):
            raw = (ph[m] / (2 * np.pi)) * lam
            branches.append(near + ((raw - near + lam / 2) % lam) - lam / 2)
        x = np.mean(branches, axis=0)
    return make_T(euler_zyx_to_R(*xi[:3]), x)
