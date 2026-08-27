"""Memory-anchored streaming stitcher -- the mechanism Pilot A measures.

Chained Sim(3) has exactly one source of information about where chunk k
belongs: chunk k-1.  Nothing ever refers back further, so per-pass error
compounds along the chain.  A pose-indexed, content-addressable memory
supplies what chaining structurally lacks: when a frame of chunk k looks
like a place stored long ago, memory proposes those earlier views as
ANCHORS, the backbone is run on chunk ∪ anchors, and the chunk is tied to
the anchors' stored global poses.  That is loop closure, and it is the
only route to bounded drift.

Per chunk k >= 1:
  1. propose anchor SITES from the new frames' descriptors and from the
     chain's predicted place; a site is an old stored view plus a stored
     partner `partner_gap` keyframes away in the same chunk;
  2. one backbone pass over chunk ∪ anchor frames (cached);
  3. verify each site: (a) its two frames' relative pose inside the pass
     agrees with memory; (b) the backbone placed the site INSIDE the
     chunk's spatial extent in pass coordinates (a site with no
     co-visibility is placed nowhere in particular);
  4. S_A = robust Sim(3) from the pass onto the stored overlap frames (the
     chained placement); S_B = the same onto the verified anchors, with
     the scale taken from S_A unless the anchors' baseline is a decent
     fraction of the chunk's spread (a 2-frame scale is noise);
  5. CONSISTENCY GATE: the closure is accepted only if the discrepancy
     D = S_B o inv(S_A), evaluated at the chunk, is inside a drift budget
     that grows with the number of chunks since the last closure.  A true
     revisit exposes the chain's drift (degrees, a fraction of a chunk
     spread); a false one exposes a random transform (tens of degrees,
     metres).  Measured on TUM fr1_room with VGGT: true closures had
     D_rot <= 4.4 deg, false ones >= 21 deg.  Without this gate every
     row was destroyed by false closures -- verifying a site against
     itself (step 3a alone) passes for any adjacent pair anywhere;
  6. correction="relax": a local pose-graph solve over the chunks since
     the last closure, earlier chunks fixed -- the amortised, streaming
     form of the batch back-end (cost O(stretch)); "distribute" spreads
     D linearly; "jump" places chunk k by S_B and moves nothing else;
     "none" keeps the chained placement (edges are still recorded);
  7. bind the chunk's new frames at their global poses.

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


def _spread(centres):
    c = np.asarray(centres, float)
    if len(c) < 2:
        return 1.0
    return float(np.median(np.linalg.norm(c - c.mean(0), axis=1))) + 1e-12


class AnchoredStitcher:
    def __init__(self, index=None, n_sites=2, recent_window=None,
                 top_proposals=5, desc_thresh=None, partner_gap=3,
                 site_rot_deg=10.0, site_dir_deg=25.0, extent_factor=3.0,
                 rot_thresh_deg=10.0, pos_thresh_rel=0.5,
                 min_old_frames=2, min_baseline_rel=1.0,
                 budget_rot=(10.0, 3.0, 45.0), budget_pos=(1.0, 0.5),
                 budget_logscale=0.5, tight_rot=(3.0, 1.0, 15.0),
                 tight_pos=(0.5, 0.15), require_appearance=True,
                 remeasure=False, remeasure_rot_deg=8.0,
                 remeasure_dir_deg=20.0, remeasure_ratio=1.5,
                 correction="relax", robust=True, pose_proposals=True,
                 smooth_junctions=True, mutual_nn=False, site_agree_rot=1e9,
                 site_agree_pos=1e9, revoke=True, verbose=False):
        self.index = index if index is not None else DescriptorIndex()
        self.n_sites = int(n_sites)
        self.recent_window = recent_window
        self.top = int(top_proposals)
        self.desc_thresh = desc_thresh
        self.partner_gap = int(partner_gap)
        self.site_rot_deg, self.site_dir_deg = site_rot_deg, site_dir_deg
        self.extent_factor = extent_factor
        self.rot_thresh_deg, self.pos_thresh_rel = rot_thresh_deg, pos_thresh_rel
        self.min_old_frames = int(min_old_frames)
        self.min_baseline_rel = min_baseline_rel
        self.budget_rot, self.budget_pos = budget_rot, budget_pos
        self.budget_logscale = budget_logscale
        self.tight_rot, self.tight_pos = tight_rot, tight_pos
        self.require_appearance = require_appearance
        self.remeasure = remeasure
        self.remeasure_rot_deg, self.remeasure_dir_deg = remeasure_rot_deg, remeasure_dir_deg
        self.remeasure_ratio = remeasure_ratio
        assert correction in ("relax", "distribute", "jump", "none")
        self.correction = correction
        self.robust = robust
        self.pose_proposals = pose_proposals
        self.smooth_junctions = smooth_junctions
        self.mutual_nn = mutual_nn
        self.site_agree_rot, self.site_agree_pos = site_agree_rot, site_agree_pos
        self.revoke = revoke
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

    def budget(self, n_stretch, n_sites=2):
        """Drift budget for a closure after n_stretch un-anchored chunks:
        rotation (deg), position (in units of the chunk's spread), and
        |log scale|.

        CONSENSUS RULE: a closure supported by ONE site may only nudge
        (the tight budget); a large correction needs two independent
        sites that agree (the loose budget).  On TUM fr1_room a single
        pose-proposed site passed the loose budget with an 11.9 deg
        "drift" after 5 chunks -- twice the whole sequence's chained loop
        error -- and moved five chunks the wrong way.
        """
        if n_sites >= 2:
            r0, r1, rmax = self.budget_rot
            p0, p1 = self.budget_pos
        else:
            r0, r1, rmax = self.tight_rot
            p0, p1 = self.tight_pos
        return (min(rmax, r0 + r1 * n_stretch), p0 + p1 * n_stretch,
                self.budget_logscale)

    def _propose_sites(self, new, descriptors, stored, owner, k, chunk_len):
        """Old anchor sites for chunk k: (j, partner, score, cos) with
        owner < k-1 and outside the recent window."""
        if self.n_sites <= 0 or not stored:
            return []
        recent = self.recent_window if self.recent_window is not None \
            else 2 * chunk_len
        lo = min(new) - recent
        exclude = {g for g in stored if g >= lo or owner[g] >= k - 1}
        cands = {}
        new_desc = np.stack([descriptors[i] for i in new])
        for i in new:
            for (j, score, cos) in self.index.propose(
                    descriptors[i], exclude=exclude, top=self.top,
                    desc_thresh=self.desc_thresh):
                if self.mutual_nn:
                    # j must also find THIS chunk among its nearest views:
                    # the query's best match must be the match's best
                    # query (perceptual aliasing -- an office of identical
                    # desks, a floor of identical boards -- fails this far
                    # more often than a true revisit does)
                    sims = new_desc @ self.index.T_desc(j)
                    if float(sims.max()) < cos - 1e-6 and \
                            int(np.argmax(sims)) != new.index(i):
                        if float(sims.max()) > cos + 0.02:
                            continue
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
            for gap in range(self.partner_gap, 0, -1):
                for p in (j + gap, j - gap):
                    if p in stored and owner[p] == oc and p not in exclude:
                        partner = p
                        break
                if partner is not None:
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

    def _verify_site(self, site, P, pos_in_pass, stored, chunk_pos):
        """(a) the site's internal relative pose agrees with memory;
        (b) the backbone placed the site inside the chunk's extent."""
        j, p = site[0], site[1]
        rel_pass = np.linalg.inv(P[pos_in_pass[j]]) @ P[pos_in_pass[p]]
        rel_st = np.linalg.inv(stored[j]) @ stored[p]
        rot = rotation_angle_deg(rel_pass[:3, :3].T @ rel_st[:3, :3])
        dr = _dir_err_deg(rel_pass[:3, 3], rel_st[:3, 3])
        cc = P[chunk_pos, :3, 3]
        sp = _spread(cc)
        dist = max(np.linalg.norm(P[pos_in_pass[g], :3, 3] - cc.mean(0))
                   for g in (j, p)) / sp
        ok = (rot < self.site_rot_deg and dr < self.site_dir_deg
              and dist < self.extent_factor)
        return ok, rot, dr, float(dist)

    def _remeasure_site(self, site, P, pos_in_pass, idx, new, descriptors,
                        cache, runner):
        """Independent check of a site: re-run the backbone on a MINIMAL
        pass -- the two chunk frames that best match the site plus the
        site's two frames -- and require the chunk->anchor relative pose
        to agree with the big pass (rotation, translation direction, and
        the scale-free distance ratio |c_q - c_j| / |c_j - c_p|).

        A co-visible anchor is placed the same way in any context; an
        anchor with nothing in common with the chunk is placed arbitrarily
        and differently in the two passes.  This is what separates a
        revisit from a look-alike, and it does not depend on how large
        the drift is.  Cost: one 4-frame pass per candidate site.
        """
        j, p = site[0], site[1]
        if descriptors is None:
            return True, dict(skipped=True)
        cos = [(float(descriptors[g] @ descriptors[j]), g) for g in idx]
        cos.sort(reverse=True)
        q = cos[0][1]
        q2 = next((g for _, g in cos[1:] if abs(g - q) <= 3), cos[1][1])
        mini = [q, q2, j, p]
        Pm = cache.get(mini, runner)["poses"]
        mpos = {g: i for i, g in enumerate(mini)}

        def rel(Pp, ip, iq, ij, ipp):
            r = np.linalg.inv(Pp[iq]) @ Pp[ij]
            d_qj = np.linalg.norm(Pp[ij, :3, 3] - Pp[iq, :3, 3])
            d_jp = np.linalg.norm(Pp[ij, :3, 3] - Pp[ipp, :3, 3]) + 1e-9
            return r, d_qj / d_jp

        r_big, ratio_big = rel(P, None, pos_in_pass[q], pos_in_pass[j], pos_in_pass[p])
        r_min, ratio_min = rel(Pm, None, mpos[q], mpos[j], mpos[p])
        rot = rotation_angle_deg(r_big[:3, :3].T @ r_min[:3, :3])
        dr = _dir_err_deg(r_big[:3, 3], r_min[:3, 3])
        ratio = max(ratio_big, ratio_min) / max(min(ratio_big, ratio_min), 1e-9)
        ok = (rot < self.remeasure_rot_deg and dr < self.remeasure_dir_deg
              and ratio < self.remeasure_ratio)
        return ok, dict(q=int(q), rot_deg=float(rot), dir_deg=float(dr),
                        ratio=float(ratio))

    # ---------------------------------------------------------------- run
    def run(self, chunks, cache, runner, descriptors=None):
        stored, local, owner = {}, {}, {}
        passes, edges, events = [], [], []
        chunk_len = max(len(c) for c in chunks)
        last_closed = 0                 # most recent chunk placed by anchors
        n_rejected = 0
        recovering = False              # after a session start, until re-anchored
        prev_scale = 1.0
        pending = None                  # last accepted closure: snapshot + evidence
        n_revoked = 0

        def bind(gi, T, k):
            stored[gi] = np.asarray(T, float)
            owner[gi] = k
            s = descriptors[gi] if descriptors is not None else np.zeros(448)
            self.index.add(gi, stored[gi], s)

        def node_of(c):
            """Current similarity chunk-c-local -> world."""
            mine = [g for g, o in owner.items() if o == c]
            if len(mine) < 2:
                return sim3.identity()
            S, _ = sim3.fit_poses(np.stack([local[g] for g in mine]),
                                  np.stack([stored[g] for g in mine]))
            return S

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
            session_start = len(overlap) < 2
            if session_start:
                # first chunk of a new session: nothing to chain from.  It
                # is placed by RELOCALISATION -- verified anchor sites only,
                # scale carried over from the previous pass (all passes
                # are regauged to unit median translation).  Until a
                # closure with two agreeing sites has been accepted, the
                # stretch is in RECOVERY: the drift budget does not apply,
                # because there is no chain to be consistent with.
                recovering = True

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
            chunk_pos = [pos[g] for g in idx]
            for gi in new:
                local[gi] = P[pos[gi]]

            # -- verify the sites
            good_old, site_log, n_ok_sites, n_app_sites = [], [], 0, 0
            for site in sites:
                ok, rot, dr, dist = self._verify_site(site, P, pos, stored,
                                                      chunk_pos)
                rem = None
                if ok and self.remeasure:
                    ok, rem = self._remeasure_site(site, P, pos, idx, new,
                                                   descriptors, cache, runner)
                site_log.append(dict(view=int(site[0]), partner=int(site[1]),
                                     owner=int(owner[site[0]]), score=site[2],
                                     cos=site[3], rot_deg=rot, dir_deg=dr,
                                     extent=dist, remeasure=rem, ok=bool(ok)))
                if ok:
                    good_old += [site[0], site[1]]
                    n_ok_sites += 1
                    n_app_sites += int(site[3] > 0.0)
            if self.require_appearance and n_app_sites == 0:
                good_old = []            # pose-only sites cannot close a loop

            # -- chained placement from the overlap
            if not session_start:
                A_ov = P[[pos[g] for g in overlap]]
                B_ov = np.stack([stored[g] for g in overlap])
                S_A, keep_ov, info_ov = self._fit(A_ov, B_ov)
            else:
                # no chain: continue from the last stored pose with the
                # previous scale (a placeholder until anchors place it)
                last = max(stored)
                S_A = (prev_scale, stored[last][:3, :3] @ P[chunk_pos[0], :3, :3].T,
                       stored[last][:3, 3] - prev_scale * (stored[last][:3, :3] @ P[chunk_pos[0], :3, :3].T @ P[chunk_pos[0], :3, 3]))
                keep_ov = np.zeros(0, bool)
            S, loop = S_A, None
            world_chunk = sim3.apply(S_A, P[chunk_pos])
            spread_w = _spread(world_chunk[:, :3, 3])
            centroid_pass = P[chunk_pos, :3, 3].mean(0)

            # -- loop placement from verified old anchors, then the gate
            if n_ok_sites >= 2 and len(good_old) >= 4:
                # per-site placements must agree with each other
                per_site = []
                for s_i in range(0, len(good_old), 2):
                    fr = good_old[s_i:s_i + 2]
                    Si = sim3.fit_poses_fixed_scale(
                        P[[pos[g] for g in fr]], np.stack([stored[g] for g in fr]), S_A[0])
                    per_site.append(Si)
                if len(per_site) >= 2:
                    Da = sim3.compose(per_site[0], sim3.inverse(per_site[1]))
                    rot_d = rotation_angle_deg(Da[1])
                    pa = per_site[0][0] * (per_site[0][1] @ centroid_pass) + per_site[0][2]
                    pb = per_site[1][0] * (per_site[1][1] @ centroid_pass) + per_site[1][2]
                    pos_d = float(np.linalg.norm(pa - pb)) / spread_w
                    if rot_d > self.site_agree_rot or pos_d > self.site_agree_pos:
                        # the sites disagree -- the aliasing signature.  Keep
                        # the one closer to dead reckoning (S_A) but demote
                        # the closure to single-site, i.e. the tight budget:
                        # a lone survivor may only nudge.
                        pA = S_A[0] * (S_A[1] @ centroid_pass) + S_A[2]
                        dev = []
                        for Si in per_site:
                            pi_ = Si[0] * (Si[1] @ centroid_pass) + Si[2]
                            Di = sim3.compose(Si, sim3.inverse(S_A))
                            dev.append(rotation_angle_deg(Di[1]) / 10.0
                                       + np.linalg.norm(pi_ - pA) / spread_w)
                        keep_i = int(np.argmin(dev))
                        # Is the discarded site plain garbage (the robust
                        # 4-frame fit already marks its frames as outliers)
                        # or a plausible look-alike (internally consistent,
                        # just elsewhere)?  Garbage: drop it, keep the loose
                        # budget.  Look-alike: keep the closer site under
                        # the tight budget -- a lone survivor may only nudge.
                        _, keep4, _ = self._fit(P[[pos[g] for g in good_old]],
                                                np.stack([stored[g] for g in good_old]))
                        drop = 1 - keep_i
                        garbage = not keep4[2 * drop: 2 * drop + 2].any()
                        ok_sites = [s for s in site_log if s["ok"]]
                        for s_i, s in enumerate(ok_sites):
                            if s_i != keep_i:
                                s["ok"] = False
                                s["disagree"] = dict(rot_deg=float(rot_d), pos_rel=pos_d,
                                                     garbage=bool(garbage))
                        good_old = good_old[2 * keep_i: 2 * keep_i + 2]
                        n_app_sites = int(ok_sites[keep_i]["cos"] > 0.0)
                        n_ok_sites = 2 if garbage else 1
            if len(good_old) >= self.min_old_frames:
                A_old = P[[pos[g] for g in good_old]]
                B_old = np.stack([stored[g] for g in good_old])
                S_B, keep_old, info_old = self._fit(A_old, B_old)
                inl = [g for g, kp in zip(good_old, keep_old) if kp]
                base = (max(np.linalg.norm(stored[g][:3, 3] - stored[h][:3, 3])
                            for g in inl for h in inl) if len(inl) >= 2 else 0.0)
                baseline_rel = base / spread_w
                scale_from_anchors = baseline_rel >= self.min_baseline_rel \
                    and info_old["scale_ok"]
                if len(inl) >= self.min_old_frames:
                    if not scale_from_anchors:      # scale is noise: take S_A's
                        S_B = sim3.fit_poses_fixed_scale(
                            P[[pos[g] for g in inl]],
                            np.stack([stored[g] for g in inl]), S_A[0])
                    D = sim3.compose(S_B, sim3.inverse(S_A))
                    loop_note = None
                    a_min = min(owner[g] for g in inl)
                    lo = max(a_min, last_closed)
                    n_stretch = max(1, k - lo)
                    b_rot, b_pos, b_ls = self.budget(n_stretch, n_ok_sites)
                    d_rot = rotation_angle_deg(D[1])
                    pA = S_A[0] * (S_A[1] @ centroid_pass) + S_A[2]
                    pB = S_B[0] * (S_B[1] @ centroid_pass) + S_B[2]
                    d_pos = float(np.linalg.norm(pA - pB)) / spread_w
                    d_ls = float(abs(np.log(S_B[0] / S_A[0])))
                    ok = (d_rot <= b_rot and d_pos <= b_pos
                          and (d_ls <= b_ls or not scale_from_anchors))
                    if recovering and n_ok_sites >= 2:
                        ok = True             # relocalisation: no chain to agree with
                    reason = None if ok else (
                        "rot" if d_rot > b_rot else
                        "pos" if d_pos > b_pos else "scale")
                    loop = dict(anchor_chunk=int(a_min), n_old=len(inl),
                                n_sites=int(n_ok_sites), n_app_sites=int(n_app_sites),
                                relocalisation=bool(recovering), note=loop_note,
                                D_rot_deg=d_rot, D_pos_rel=d_pos, D_logscale=d_ls,
                                baseline_rel=float(baseline_rel),
                                scale_from_anchors=bool(scale_from_anchors),
                                n_stretch=int(n_stretch),
                                budget=dict(rot=b_rot, pos=b_pos, logscale=b_ls),
                                accepted=bool(ok), reason=reason,
                                stretch=[int(lo), int(k)],
                                applied=self.correction if ok else "rejected")
                    # REVOCATION.  The closure accepted at the previous chunk is
                    # provisional.  If THIS chunk brings stronger evidence (>= 2
                    # verified sites) that disagrees with the chain by more than
                    # twice the budget, and the previous closure rested on one
                    # site, the previous closure was the look-alike: restore the
                    # snapshot taken before it, re-fit the chain, and judge this
                    # closure against the restored chain.  (Streaming counterpart
                    # of the batch outlier rejection; one wrong closure on TUM
                    # floor / room made every later chunk inherit its error.)
                    if (self.revoke and not ok and pending is not None
                            and pending["k"] == k - 1 and n_ok_sites >= 2
                            and pending["n_sites"] <= 1
                            and (d_rot > 2 * b_rot or d_pos > 2 * b_pos)):
                        for g, T in pending["snapshot"].items():
                            stored[g] = T
                            self.index.update_pose(g, T)
                        edges = [e for e in edges if e["k"] != pending["k"] or e["kind"] != "loop"]
                        n_revoked += 1
                        events[pending["k"]]["loop"]["revoked_at"] = int(k)
                        # re-fit the chain on the restored overlap and re-judge
                        A_ov = P[[pos[g] for g in overlap]]
                        B_ov = np.stack([stored[g] for g in overlap])
                        S_A, keep_ov, info_ov = self._fit(A_ov, B_ov)
                        pA = S_A[0] * (S_A[1] @ centroid_pass) + S_A[2]
                        D = sim3.compose(S_B, sim3.inverse(S_A))
                        d_rot = rotation_angle_deg(D[1])
                        d_pos = float(np.linalg.norm(pA - pB)) / spread_w
                        d_ls = float(abs(np.log(S_B[0] / S_A[0])))
                        ok = (d_rot <= b_rot and d_pos <= b_pos
                              and (d_ls <= b_ls or not scale_from_anchors))
                        reason = None if ok else ("rot" if d_rot > b_rot else "pos" if d_pos > b_pos else "scale")
                        loop_note = "after revocation"
                        pending = None
                    if not ok:
                        n_rejected += 1
                        good_old = []            # a rejected closure is no edge
                    else:
                        loop_edge_frames = inl
                        # a two-frame scale is only as good as its baseline:
                        # full weight from two chunk spreads upward
                        loop_w_scale = (min(1.0, 0.5 * float(baseline_rel))
                                        if scale_from_anchors else 0.0)

            # -- edges for the batch solver (identical measurements)
            by_owner = {}
            for g in list(overlap) + list(good_old):
                by_owner.setdefault(owner[g], []).append(g)
            new_edges = []
            for c, frames in by_owner.items():
                if len(frames) < 2:
                    continue
                Z, keep_e, info_e = self._fit(P[[pos[g] for g in frames]],
                                              np.stack([local[g] for g in frames]))
                if info_e["scale_ok"]:
                    kind = "seq" if c == k - 1 else "loop"
                    new_edges.append(dict(c=int(c), k=int(k), Z=Z,
                                          n=int(keep_e.sum()), kind=kind,
                                          w_scale=1.0 if kind == "seq" else loop_w_scale))
            edges += new_edges

            # -- placement / correction
            if loop and loop["accepted"]:
                pending = dict(k=k, n_sites=loop["n_sites"],
                               snapshot={g: T.copy() for g, T in stored.items()})
            if loop and loop["accepted"] and recovering:
                S = S_B                       # relocalised: nothing to relax
                last_closed = k
                recovering = False
            elif loop and loop["accepted"]:
                lo, a_min = loop["stretch"][0], loop["anchor_chunk"]
                if self.correction == "relax":
                    from .posegraph import solve_nodes
                    nodes = {c: node_of(c) for c in range(k)}
                    nodes[k] = S_A
                    free = [c for c in range(lo + 1, k + 1)]
                    sub = [e for e in edges if e["k"] <= k and
                           (e["c"] in free or e["k"] in free)]
                    sol = solve_nodes(nodes, sub, free)
                    for c in free:
                        if c == k:
                            continue
                        for g, o in owner.items():
                            if o == c:
                                stored[g] = sim3.apply(sol[c], local[g][None])[0]
                                self.index.update_pose(g, stored[g])
                    S = sol[k]
                    last_closed = k
                elif self.correction == "distribute":
                    D = sim3.compose(S_B, sim3.inverse(S_A))
                    if k - lo >= 2:
                        for g, c in owner.items():
                            if lo < c < k:
                                C = sim3.interpolate(D, (c - lo) / (k - lo))
                                stored[g] = sim3.apply(C, stored[g][None])[0]
                                self.index.update_pose(g, stored[g])
                    S = S_B
                    last_closed = k
                elif self.correction == "jump":
                    S = S_B
                    last_closed = k
                # "none": chained placement, edge recorded

            moved = sim3.apply(S, P)
            prev_scale = float(S[0])
            for gi in new:
                bind(gi, moved[pos[gi]], k)
            if self.smooth_junctions and loop and loop["accepted"]:
                # Relaxing whole chunks leaves a step at the junction: the
                # overlap frames sit where chunk k-1 put them while chunk
                # k's new frames sit where the closure put them (measured
                # on TUM fr1_room: AUC_in 80.8 -> 75.8).  Blend each overlap
                # frame between its owner's placement and chunk k's, first
                # frame staying with the owner, last frame almost with k.
                n_ov = len(overlap)
                for r_i, g in enumerate(overlap):
                    w = (r_i + 1) / (n_ov + 1)
                    rel = np.linalg.inv(stored[g]) @ moved[pos[g]]
                    C = sim3.interpolate((1.0, rel[:3, :3], rel[:3, 3]), w)
                    T = stored[g] @ np.block([[C[1], C[2][:, None]], [np.zeros((1, 3)), 1.0]])
                    stored[g] = T
                    self.index.update_pose(g, T)
            ev = dict(chunk=k, n_new=len(new), n_overlap=len(overlap),
                      n_overlap_inliers=int(keep_ov.sum()), sites=site_log,
                      loop=loop, session_start=bool(session_start),
                      recovering=bool(recovering))
            events.append(ev)
            if self.verbose:
                if loop and loop["accepted"]:
                    tag = (f"LOOP->chunk {loop['anchor_chunk']} D_rot "
                           f"{loop['D_rot_deg']:.2f} deg D_pos {loop['D_pos_rel']:.2f} "
                           f"spread ({self.correction})")
                elif loop:
                    tag = (f"closure REJECTED ({loop['reason']}): D_rot "
                           f"{loop['D_rot_deg']:.1f} deg D_pos {loop['D_pos_rel']:.2f} "
                           f"D_logs {loop['D_logscale']:.2f} vs budget "
                           f"{loop['budget']['rot']:.0f}/{loop['budget']['pos']:.2f}/"
                           f"{loop['budget']['logscale']:.2f}")
                else:
                    tag = "chain"
                print(f"    chunk {k:>3}: pass {len(pass_idx)} frames, "
                      f"{len(sites)} sites ({sum(s['ok'] for s in site_log)} verified), {tag}")

        order = sorted(stored)
        est = np.stack([stored[i] for i in order])
        return dict(est=est, passes=passes, edges=edges, events=events,
                    owner=[int(owner[i]) for i in order],
                    local={int(g): local[g] for g in order},
                    n_loops=sum(1 for e in events if e.get("loop") and e["loop"]["accepted"]
                                and "revoked_at" not in e["loop"]),
                    n_rejected=n_rejected, n_revoked=n_revoked)


def stitch_chained(chunks, cache, runner, robust=True):
    """Plain sequential Sim(3) chaining, written independently of the
    stitcher above so the n_sites=0 reduction can be tested against it."""
    glob, prev_scale = {}, 1.0
    for k, idx in enumerate(chunks):
        P = cache.get(idx, runner)["poses"]
        if k == 0:
            for li, gi in enumerate(idx):
                glob[gi] = P[li]
            continue
        ov = [(li, gi) for li, gi in enumerate(idx) if gi in glob]
        if len(ov) < 2:
            # session start: a chain has nothing to align to.  It continues
            # from its last pose with its previous scale -- the honest
            # definition of the baseline on multi-session data (the memory-
            # anchored stitcher relocalises here instead).
            last = max(glob)
            R = glob[last][:3, :3] @ P[0, :3, :3].T
            S = (prev_scale, R, glob[last][:3, 3] - prev_scale * (R @ P[0, :3, 3]))
        else:
            A = P[[li for li, _ in ov]]
            B = np.stack([glob[gi] for _, gi in ov])
            if robust and len(A) >= 3:
                S, _, _ = sim3.fit_poses_robust(A, B, min_inliers=3)
            else:
                S, _ = sim3.fit_poses(A, B)
        prev_scale = float(S[0])
        moved = sim3.apply(S, P)
        for li, gi in enumerate(idx):
            glob.setdefault(gi, moved[li])
    return np.stack([glob[i] for i in sorted(glob)])
