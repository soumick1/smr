"""Memory-anchored streaming stitcher -- the mechanism Pilot A measures.

Chained Sim(3) has exactly one source of information about where chunk k
belongs: chunk k-1.  Nothing ever refers back further, so per-pass gauge
error compounds along the chain.  A pose-indexed, content-addressable
memory supplies what chaining structurally lacks: when a frame of chunk k
looks like a place stored long ago, memory proposes those earlier views as
ANCHORS, the backbone is run on chunk ∪ anchors, and the chunk is placed
against the anchors' stored global poses.  That is loop closure, and it
is the only route to bounded drift.

Per chunk k >= 1:
  1. propose anchor SITES from the new frames' descriptors (index.propose);
     a site is an old stored view plus its stored temporal neighbour, so
     every site can be verified geometrically and contributes scale;
  2. one backbone pass over chunk ∪ anchor frames (cached);
  3. verify each site: the relative pose of the pair inside the pass must
     agree with the pair's stored relative pose (rotation / direction);
  4. S_A = robust Sim(3) from the pass onto the stored poses of the
     overlap frames (the chained placement);
     S_B = the same onto the VERIFIED old-anchor frames (the loop
     placement); when S_B exists the chunk is placed by S_B and the
     discrepancy D = S_B o inv(S_A) is the drift the loop just exposed;
  5. correction="distribute": D is spread over the stored frames between
     the anchor chunk and chunk k (identity at the anchor chunk, D at
     chunk k-1) -- the amortised, one-shot, streaming form of pose-graph
     relaxation; "jump": earlier frames are left where they were;
  6. bind the chunk's new frames at their global poses.

With n_sites=0 no anchor is ever proposed and the pass is the bare chunk,
so the SAME code path produces the chained baseline (tested).  Every
pass and every accepted edge is recorded so a batch pose-graph solve can
be run on identical measurements (posegraph.py).
"""
from __future__ import annotations

import numpy as np

from ..eval.trajectory import rotation_angle_deg
from . import sim3
from .memory_index import DescriptorIndex


def _dir_err_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(a @ b) / (na * nb), -1, 1))))


class AnchoredStitcher:
    def __init__(self, index=None, n_sites=2, recent_window=None,
                 top_proposals=5, desc_thresh=None,
                 site_rot_deg=10.0, site_dir_deg=25.0,
                 rot_thresh_deg=10.0, pos_thresh_rel=0.5,
                 min_old_frames=2, correction="distribute", robust=True,
                 pose_proposals=True, verbose=False):
        self.index = index if index is not None else DescriptorIndex()
        self.n_sites = int(n_sites)
        self.recent_window = recent_window
        self.top = int(top_proposals)
        self.desc_thresh = desc_thresh
        self.site_rot_deg, self.site_dir_deg = site_rot_deg, site_dir_deg
        self.rot_thresh_deg, self.pos_thresh_rel = rot_thresh_deg, pos_thresh_rel
        self.min_old_frames = int(min_old_frames)
        assert correction in ("distribute", "jump", "none")
        self.correction = correction
        self.robust = robust
        self.pose_proposals = pose_proposals
        self.verbose = verbose

    # ------------------------------------------------------------- helpers
    def _fit(self, A, B):
        if self.robust and len(A) >= 3:
            S, keep, info = sim3.fit_poses_robust(
                A, B, rot_thresh_deg=self.rot_thresh_deg,
                pos_thresh_rel=self.pos_thresh_rel, min_inliers=3)
        else:
            S, info = sim3.fit_poses(A, B)
            keep = np.ones(len(A), bool)
        return S, keep, info

    def _propose_sites(self, new, descriptors, stored, owner, k, chunk_len):
        """Old anchor sites for chunk k: (j, partner) pairs with owner < k-1
        and outside the recent window."""
        if self.n_sites <= 0 or not stored:
            return []
        recent = self.recent_window if self.recent_window is not None \
            else 2 * chunk_len
        lo = min(new) - recent
        exclude = {g for g in stored if g >= lo or owner[g] >= k - 1}
        cands = {}
        for i in new:
            for (j, score, cos) in self.index.propose(
                    descriptors[i], exclude=exclude, top=self.top,
                    desc_thresh=self.desc_thresh):
                if score > cands.get(j, (-np.inf, 0.0))[0]:
                    cands[j] = (score, cos)
        if self.pose_proposals and new:
            # where the chain would put this chunk: near the last stored
            # frame; any OLD view stored near there is a candidate too
            last = max(stored)
            for j in self.index.near_pose(stored[last], k=self.top,
                                          exclude=exclude):
                cands.setdefault(j, (0.0, 0.0))
        sites, used_chunks = [], set()
        for j in sorted(cands, key=lambda g: -cands[g][0]):
            oc = owner[j]
            partner = None
            for p in (j + 1, j - 1, j + 2, j - 2):
                if p in stored and owner[p] == oc and p not in exclude:
                    partner = p
                    break
            if partner is None:
                continue
            if oc in used_chunks and len(used_chunks) < self.n_sites:
                continue                      # diversify across old chunks
            sites.append((j, partner, cands[j][0], cands[j][1]))
            used_chunks.add(oc)
            if len(sites) >= self.n_sites:
                break
        return sites

    def _verify_site(self, site, P, pos_in_pass, stored):
        j, p = site[0], site[1]
        rel_pass = np.linalg.inv(P[pos_in_pass[j]]) @ P[pos_in_pass[p]]
        rel_st = np.linalg.inv(stored[j]) @ stored[p]
        rot = rotation_angle_deg(rel_pass[:3, :3].T @ rel_st[:3, :3])
        dr = _dir_err_deg(rel_pass[:3, 3], rel_st[:3, 3])
        ok = rot < self.site_rot_deg and dr < self.site_dir_deg
        return ok, rot, dr

    # ---------------------------------------------------------------- run
    def run(self, chunks, cache, runner, descriptors=None):
        stored, local, owner = {}, {}, {}
        passes, edges, events = [], [], []
        chunk_len = max(len(c) for c in chunks)
        last_closed = 0                 # most recent chunk placed by anchors

        def bind(gi, T, k):
            stored[gi] = np.asarray(T, float)
            owner[gi] = k
            s = descriptors[gi] if descriptors is not None else np.zeros(448)
            self.index.add(gi, stored[gi], s)

        for k, idx in enumerate(chunks):
            new = [i for i in idx if i not in stored]
            overlap = [i for i in idx if i in stored]
            if k == 0:
                P = cache.get(idx, runner)["poses"]
                passes.append(list(idx))
                for li, gi in enumerate(idx):
                    local[gi] = P[li]
                    bind(gi, P[li], 0)
                events.append(dict(chunk=0, n_new=len(idx)))
                continue
            assert len(overlap) >= 2, (
                f"chunk {k} shares only {len(overlap)} frames with memory; "
                f"increase --overlap")

            sites = self._propose_sites(new, descriptors, stored, owner, k,
                                        chunk_len) if descriptors is not None \
                else []
            anchors = []
            for (j, p, _, _) in sites:
                for g in (j, p):
                    if g not in idx and g not in anchors:
                        anchors.append(g)
            pass_idx = list(idx) + anchors
            P = cache.get(pass_idx, runner)["poses"]
            passes.append(pass_idx)
            pos = {gi: li for li, gi in enumerate(pass_idx)}
            for gi in new:
                local[gi] = P[pos[gi]]

            # -- verify the sites
            good_old, site_log = [], []
            for site in sites:
                ok, rot, dr = self._verify_site(site, P, pos, stored)
                site_log.append(dict(view=int(site[0]), partner=int(site[1]),
                                     owner=int(owner[site[0]]), score=site[2],
                                     cos=site[3], rot_deg=rot, dir_deg=dr,
                                     ok=bool(ok)))
                if ok:
                    good_old += [site[0], site[1]]

            # -- chained placement from the overlap
            A_ov = P[[pos[g] for g in overlap]]
            B_ov = np.stack([stored[g] for g in overlap])
            S_A, keep_ov, info_ov = self._fit(A_ov, B_ov)
            S, loop = S_A, None

            # -- loop placement from verified old anchors
            if len(good_old) >= self.min_old_frames:
                A_old = P[[pos[g] for g in good_old]]
                B_old = np.stack([stored[g] for g in good_old])
                S_B, keep_old, info_old = self._fit(A_old, B_old)
                inl = [g for g, kp in zip(good_old, keep_old) if kp]
                if len(inl) >= self.min_old_frames and info_old["scale_ok"]:
                    D = sim3.compose(S_B, sim3.inverse(S_A))
                    a_min = min(owner[g] for g in inl)
                    loop = dict(anchor_chunk=int(a_min), n_old=len(inl),
                                D_rot_deg=rotation_angle_deg(D[1]),
                                D_trans=float(np.linalg.norm(D[2])),
                                D_logscale=float(np.log(D[0])),
                                applied=self.correction)
                    # Distribute D only over the UN-ANCHORED stretch: the
                    # chunks since the last closure (or since the anchor
                    # chunk, whichever is later).  D is measured inside one
                    # pass, so it carries that pass's own distortion between
                    # the overlap frames and the anchors; if chunk k-1 was
                    # itself anchored there is no chain drift to remove and
                    # spreading D would corrupt correct placements (seen in
                    # simulation: closures after the first one re-measure
                    # within-pass distortion, not drift).
                    lo = max(a_min, last_closed)
                    loop["stretch"] = [int(lo), int(k)]
                    if self.correction == "distribute" and k - lo >= 2:
                        for g, c in owner.items():
                            if lo < c < k:
                                alpha = (c - lo) / (k - lo)
                                C = sim3.interpolate(D, alpha)
                                stored[g] = sim3.apply(C, stored[g][None])[0]
                                self.index.update_pose(g, stored[g])
                    if self.correction in ("distribute", "jump"):
                        S = S_B
                        last_closed = k
                    # correction == "none": keep the chained placement, but
                    # still record the loop edge for the batch solver
            # -- edges for the batch solver (identical measurements)
            by_owner = {}
            for g in list(overlap) + list(good_old):
                by_owner.setdefault(owner[g], []).append(g)
            for c, frames in by_owner.items():
                if len(frames) < 2:
                    continue
                Z, keep_e, info_e = self._fit(P[[pos[g] for g in frames]],
                                              np.stack([local[g] for g in frames]))
                if info_e["scale_ok"]:
                    edges.append(dict(c=int(c), k=int(k), Z=Z, n=int(keep_e.sum()),
                                      kind="seq" if c == k - 1 else "loop"))

            moved = sim3.apply(S, P)
            for gi in new:
                bind(gi, moved[pos[gi]], k)
            ev = dict(chunk=k, n_new=len(new), n_overlap=len(overlap),
                      n_overlap_inliers=int(keep_ov.sum()), sites=site_log,
                      loop=loop)
            events.append(ev)
            if self.verbose:
                tag = f"loop->chunk {loop['anchor_chunk']} D_rot {loop['D_rot_deg']:.2f} deg" \
                    if loop else "chain"
                print(f"    chunk {k:>3}: pass {len(pass_idx)} frames, "
                      f"{len(sites)} sites, {tag}")

        order = sorted(stored)
        est = np.stack([stored[i] for i in order])
        return dict(est=est, passes=passes, edges=edges, events=events,
                    owner=[int(owner[i]) for i in order],
                    local={int(g): local[g] for g in order},
                    n_loops=sum(1 for e in events if e.get("loop")))


def stitch_chained(chunks, cache, runner, robust=True):
    """Plain sequential Sim(3) chaining, written independently of the
    stitcher above so the n_sites=0 reduction can be tested against it."""
    glob = {}
    for k, idx in enumerate(chunks):
        P = cache.get(idx, runner)["poses"]
        if k == 0:
            for li, gi in enumerate(idx):
                glob[gi] = P[li]
            continue
        ov = [(li, gi) for li, gi in enumerate(idx) if gi in glob]
        A = P[[li for li, _ in ov]]
        B = np.stack([glob[gi] for _, gi in ov])
        if robust and len(A) >= 3:
            S, _, _ = sim3.fit_poses_robust(A, B, min_inliers=3)
        else:
            S, _ = sim3.fit_poses(A, B)
        moved = sim3.apply(S, P)
        for li, gi in enumerate(idx):
            glob.setdefault(gi, moved[li])
    return np.stack([glob[i] for i in sorted(glob)])
