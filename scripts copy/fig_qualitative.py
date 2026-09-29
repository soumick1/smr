#!/usr/bin/env python3
"""Qualitative figures for the camera-pose and SLAM rows, from a pilot_a run.

    python scripts/fig_qualitative.py --est outputs/est/co3d_teddy_vggt_n200.npz \
        --json outputs/reports/pilotA_co3d_teddy_vggt_s1_c16.json --out outputs/figures/qual_co3d_teddy

Inputs: the `--save-est` npz of experiments/pilot_a.py (rows' trajectories, keyframe indices, chunks), its `--json`
report (closure events), and the GT npz it names (poses).  Produces <out>.pdf/.png with three panels:
  (a) pairwise relative-pose error matrices (deg; max of rotation and translation-direction error, the AUC@30
      criterion) for the raw chained row and the +SMR row, same colour scale, window boundaries marked --
      the seam bleed is the off-diagonal blocks, and the memory's effect is where those blocks fade;
  (b) top-down trajectories after Sim(3) alignment to GT (GT black, raw red, +SMR blue, +SMR+PGO dashed),
      accepted closures drawn as arcs between the closing window and its remembered site;
  (c) per-frame absolute position error for each row.
Only numpy + matplotlib; nothing here runs a backbone.
"""
import argparse, json, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smr.eval.trajectory import align_to_gt  # noqa: E402


def rel_errors(P, Q):
    """Pairwise max(rotation error, translation-direction error) in degrees between the relative poses of two
    trajectories P (est) and Q (gt), each (K,4,4) c2w.  Vectorised over all pairs."""
    R_p, t_p = P[:, :3, :3], P[:, :3, 3]
    R_q, t_q = Q[:, :3, :3], Q[:, :3, 3]
    # relative rotations R_i^T R_j and translations R_i^T (t_j - t_i)
    Rrel_p = np.einsum("iab,jac->ijbc", R_p, R_p)                 # (K,K,3,3) = R_i^T R_j
    Rrel_q = np.einsum("iab,jac->ijbc", R_q, R_q)
    dR = np.einsum("ijab,ijcb->ijac", Rrel_p, Rrel_q)              # Rrel_p Rrel_q^T
    tr = np.clip((np.trace(dR, axis1=2, axis2=3) - 1) / 2, -1, 1)
    rot = np.degrees(np.arccos(tr))
    trel_p = np.einsum("iab,ija->ijb", R_p, t_p[None] - t_p[:, None])
    trel_q = np.einsum("iab,ija->ijb", R_q, t_q[None] - t_q[:, None])
    nrm = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    cos = np.clip((nrm(trel_p) * nrm(trel_q)).sum(-1), -1, 1)
    trans = np.degrees(np.arccos(cos))
    err = np.maximum(rot, trans)
    np.fill_diagonal(err, 0.0)
    return err


def closures_from_report(report, chunks, row="smr"):
    """(window_k, anchor_frame, chunk_frame) for accepted, unrevoked closures.  The stitcher logs, per window,
    `loop` (accepted, anchor_chunk, ...) and `sites` = [{view, partner, ok, ...}]: the remembered view of the
    first verified site is the anchor; if no site is logged, the middle frame of loop['anchor_chunk'] is used."""
    out = []
    events = report.get("events", {})
    if isinstance(events, dict):
        ev_list = events.get(row) or events.get("smr") or events.get("smr_pgo") or next(iter(events.values()), [])
    else:
        ev_list = events
    for ev in ev_list or []:
        loop = ev.get("loop") if isinstance(ev, dict) else None
        if not loop or not loop.get("accepted") or "revoked_at" in loop:
            continue
        k = int(ev.get("chunk", -1))
        if k < 0 or k >= len(chunks):
            continue
        anchor = None
        for s in (ev.get("sites") or loop.get("sites") or []):
            if isinstance(s, dict) and s.get("ok", True):
                for key in ("view", "anchor", "frame", "g"):
                    if key in s:
                        anchor = int(s[key]); break
            if anchor is not None:
                break
        if anchor is None and "anchor_chunk" in loop and 0 <= int(loop["anchor_chunk"]) < len(chunks):
            ac = chunks[int(loop["anchor_chunk"])]; anchor = int(ac[len(ac) // 2])
        if anchor is not None:
            out.append((k, anchor, int(chunks[k][len(chunks[k]) // 2])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True); ap.add_argument("--json", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--raw-row", default="chained"); ap.add_argument("--smr-row", default="smr"); ap.add_argument("--pgo-row", default="smr_pgo")
    ap.add_argument("--vmax", type=float, default=30.0, help="colour scale ceiling (deg) for the error matrices")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    est = np.load(a.est, allow_pickle=True)
    gt_all = np.asarray(np.load(str(est["gt"]), allow_pickle=True)["poses"], float)
    key = np.asarray(est["key"], int); gt = gt_all[key]
    chunks = [np.asarray(c, int) for c in est["chunks"]]
    rows = {str(r): np.asarray(est[f"est_{r}"], float) for r in est["rows"]}
    raw, smr = rows[a.raw_row], rows[a.smr_row]; pgo = rows.get(a.pgo_row)
    report = json.load(open(a.json)) if a.json and pathlib.Path(a.json).exists() else {}
    closures = closures_from_report(report, chunks, a.smr_row)
    # chunks and closure frames are keyframe POSITIONS (indices into the est rows / `key`), not GT frame ids
    pos_of = {i: i for i in range(len(key))}

    fig = plt.figure(figsize=(10.2, 3.7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.32)
    # (a) error matrices
    E = {"raw": rel_errors(raw, gt), "+SMR": rel_errors(smr, gt)}
    bounds = [int(c[0]) for c in chunks[1:]]
    for j, (name, M) in enumerate(E.items()):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(M, vmin=0, vmax=a.vmax, cmap="magma_r", interpolation="nearest")
        for b in bounds:
            ax.axhline(b - 0.5, color="w", lw=0.3, alpha=0.6); ax.axvline(b - 0.5, color="w", lw=0.3, alpha=0.6)
        cross = np.ones_like(M, bool)
        for c in chunks:
            idx = [int(g) for g in c]
            cross[np.ix_(idx, idx)] = False
        xw = f"cross-window mean {M[cross].mean():.1f}$^\\circ$" if cross.any() else "single window (no cross-window pairs)"
        ax.set_title(f"{name}\npairwise error; {xw}", fontsize=8)
        ax.set_xlabel("frame", fontsize=8); ax.set_ylabel("frame" if j == 0 else "", fontsize=8); ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=[fig.axes[0], fig.axes[1]], fraction=0.02, pad=0.01); cb.set_label("deg", fontsize=7); cb.ax.tick_params(labelsize=7)
    # (b) top-down trajectories
    ax = fig.add_subplot(gs[0, 2])
    G = gt[:, :3, 3]
    up = np.argmin(G.var(0))                                     # the flattest axis is "up" -> drop it
    keep = [i for i in range(3) if i != up]
    def xy(P): return P[..., keep]
    ax.plot(*xy(G).T, color="k", lw=1.2, label="GT")
    A_raw, _ = align_to_gt(raw, gt); A_smr, _ = align_to_gt(smr, gt)
    ax.plot(*xy(A_raw[:, :3, 3]).T, color="tab:red", lw=1.0, alpha=0.9, label="raw (chained windows)")
    ax.plot(*xy(A_smr[:, :3, 3]).T, color="tab:blue", lw=1.0, alpha=0.9, label="+SMR")
    A_pgo = None
    if pgo is not None:
        A_pgo, _ = align_to_gt(pgo, gt); ax.plot(*xy(A_pgo[:, :3, 3]).T, color="tab:blue", lw=0.8, ls="--", alpha=0.8, label="+SMR+PGO")
    for k, anc, cur in closures:
        if anc in pos_of and cur in pos_of:
            p, q = xy(A_smr[pos_of[anc], :3, 3]), xy(A_smr[pos_of[cur], :3, 3])
            ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-", color="tab:green", lw=0.9, alpha=0.85,
                        connectionstyle="arc3,rad=0.3"))
    ax.scatter(*xy(G[:1]).T, color="k", s=12, zorder=5)
    ax.set_aspect("equal"); ax.tick_params(labelsize=7); ax.set_title(f"trajectories (top-down); {len(closures)} accepted closures", fontsize=8)
    ax.legend(fontsize=6.5, loc="best", frameon=False)
    if a.title:
        fig.suptitle(a.title, fontsize=9)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight"); fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    # second figure: per-frame error alone
    fig = plt.figure(figsize=(4.6, 3.0))
    ax = fig.add_subplot(111)
    # (c) per-frame position error
    for name, A, col, ls in (("raw", A_raw, "tab:red", "-"), ("+SMR", A_smr, "tab:blue", "-")) + ((("+SMR+PGO", A_pgo, "tab:blue", "--"),) if A_pgo is not None else ()):
        e = np.linalg.norm(A[:, :3, 3] - G, axis=1)
        ax.plot(e, color=col, ls=ls, lw=0.9, label=f"{name}: ATE {np.sqrt((e**2).mean()):.3f}")
    for b in bounds:
        ax.axvline(b, color="0.85", lw=0.5)
    for k, anc, cur in closures:
        if cur in pos_of: ax.axvline(pos_of[cur], color="tab:green", lw=0.7, alpha=0.8)
    ax.set_xlabel("keyframe", fontsize=8); ax.set_ylabel("position error (GT units)", fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title("per-frame error; accepted closures in green", fontsize=8); ax.legend(fontsize=6.5, frameon=False)
    fig.savefig(str(out) + "_error.pdf", bbox_inches="tight"); fig.savefig(str(out) + "_error.png", dpi=600, bbox_inches="tight")
    summary = {n: (float(E[n][cross].mean()) if cross.any() else float("nan")) for n in E}
    print(f"wrote {out}.pdf/.png and {out}_error.pdf/.png; cross-window mean error raw {summary['raw']:.2f} -> +SMR {summary['+SMR']:.2f} deg; "
          f"closures {len(closures)}; keyframes {len(key)}, windows {len(chunks)}" + ("  [SINGLE WINDOW: raw == +SMR by construction]" if len(chunks) < 2 else ""))


if __name__ == "__main__":
    main()
