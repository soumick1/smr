#!/usr/bin/env python3
"""Mechanism plots for the paper (the review's list). One subcommand per figure; each writes <out>.pdf (vector) and
<out>.png (600 dpi). Only numpy + matplotlib; no backbone runs here.

  seams     error vs number of window junctions crossed   --est 'outputs/est/mech_*.npz'
  revisits  gain vs revisit density (one point per sequence, CO3D + RE10K + 7-Scenes) --reports 'outputs/reports/len_*_n200.json'
  settling  scaffold settling: valid / incoherent / coherent, iteration on x     (CPU experiment, no inputs)
  funnel    retrieval -> verification funnel                --reports '<pilot reports with events>' --est '<matching est npz>'
  flatbar   flat vs scaffold paired deltas                  --root outputs/ablate --split outputs/7scenes_test
  curves    AUC@30 vs N with native ceilings                --reports 'outputs/reports/len_*_n*.json'
  timeline  multi-session per-frame error with reset lines --est <multi-session est npz>
  nvshist   per-object dPSNR histogram with marked examples --nvs outputs/nvs/vggt --mark id1 id2 id3
  gpumem    GPU memory vs N (windowed vs native)            --reports 'outputs/reports/len_*_n*.json'
"""
import argparse, glob, json, pathlib, re, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"raw": "tab:red", "smr": "tab:blue", "pgo": "tab:blue", "flat": "tab:orange", "dyn": "tab:green"}


def save(fig, out):
    out = pathlib.Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight"); fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight")
    print(f"wrote {out}.pdf/.png")


# ----------------------------------------------------------------------------------------------- shared helpers
def rel_errors(P, Q):
    """Pairwise max(rotation, translation-direction) error in degrees between relative poses of est P and gt Q (K,4,4)."""
    R_p, t_p, R_q, t_q = P[:, :3, :3], P[:, :3, 3], Q[:, :3, :3], Q[:, :3, 3]
    Rrel_p = np.einsum("iab,jac->ijbc", R_p, R_p); Rrel_q = np.einsum("iab,jac->ijbc", R_q, R_q)
    dR = np.einsum("ijab,ijcb->ijac", Rrel_p, Rrel_q)
    rot = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=2, axis2=3) - 1) / 2, -1, 1)))
    trel_p = np.einsum("iab,ija->ijb", R_p, t_p[None] - t_p[:, None]); trel_q = np.einsum("iab,ija->ijb", R_q, t_q[None] - t_q[:, None])
    nrm = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    trans = np.degrees(np.arccos(np.clip((nrm(trel_p) * nrm(trel_q)).sum(-1), -1, 1)))
    E = np.maximum(rot, trans); np.fill_diagonal(E, 0.0); return E


def load_est(path):
    E = np.load(path, allow_pickle=True)
    gt = np.asarray(np.load(str(E["gt"]), allow_pickle=True)["poses"], float)[np.asarray(E["key"], int)]
    rows = {str(r): np.asarray(E[f"est_{r}"], float) for r in E["rows"]}
    chunks = [np.asarray(c, int) for c in E["chunks"]]
    return gt, rows, chunks, E


def owner_window(chunks, K):
    """first window each keyframe position belongs to (overlap frames -> the earlier window)"""
    own = np.full(K, -1)
    for k, c in enumerate(chunks):
        for i in c:
            if own[i] < 0: own[i] = k
    return own


def load_reports(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f)); rows = {r["method"]: r for r in d.get("rows", [])}
        if not rows: continue
        m = re.search(r"_n(\d+)\.json$", f); N = int(m.group(1)) if m else d.get("keyframes")
        out.append(dict(file=f, name=pathlib.Path(f).stem, dataset=str(d.get("dataset", "")), scene=str(d.get("scene", "")),
                        backbone=str(d.get("backbone", "")), N=N, rows=rows, ceiling=d.get("ceiling") or {}, events=d.get("events") or {},
                        keyframes=d.get("keyframes"), n_revisit=d.get("n_revisit_pairs", rows.get("smr", {}).get("n_revisit_pairs"))))
    return out


# ----------------------------------------------------------------------------------------------- 1. seams
def cmd_seams(a):
    files = sorted(glob.glob(a.est)); assert files, f"no est files match {a.est}"
    per = {r: {} for r in ("chained", "smr", "smr_pgo")}   # seams -> list of errors
    n_pairs = {}
    for f in files:
        gt, rows, chunks, _ = load_est(f); K = len(gt); own = owner_window(chunks, K)
        S = np.abs(own[:, None] - own[None, :]); iu = np.triu_indices(K, 1); s = S[iu]
        for r in per:
            if r not in rows: continue
            e = rel_errors(rows[r], gt)[iu]
            for k in np.unique(s):
                per[r].setdefault(int(k), []).append(e[s == k])
        for k in np.unique(s): n_pairs[int(k)] = n_pairs.get(int(k), 0) + int((s == k).sum())
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for r, lab, col, ls in (("chained", "raw (chained windows)", COL["raw"], "-"), ("smr", "+SMR", COL["smr"], "-"), ("smr_pgo", "+SMR+PGO", COL["pgo"], "--")):
        if not per[r]: continue
        ks = sorted(per[r]); med = [np.median(np.concatenate(per[r][k])) for k in ks]
        q1 = [np.percentile(np.concatenate(per[r][k]), 25) for k in ks]; q3 = [np.percentile(np.concatenate(per[r][k]), 75) for k in ks]
        ax.plot(ks, med, color=col, ls=ls, lw=1.3, marker="o", ms=2.5, label=lab)
        if r != "smr_pgo": ax.fill_between(ks, q1, q3, color=col, alpha=0.12, lw=0)
    ax.set_xlabel("window junctions between the two frames"); ax.set_ylabel("pairwise pose error (deg)\nmedian, IQR shaded")
    ax.set_yscale("log"); ax.set_title(f"{len(files)} sequences, {sum(n_pairs.values()):,} frame pairs"); ax.legend(frameon=False)
    save(fig, a.out)


# ----------------------------------------------------------------------------------------------- 2. revisits
def cmd_revisits(a):
    reps = load_reports(a.reports); assert reps, "no reports"
    fig, ax = plt.subplots(figsize=(3.6, 2.6)); mk = {"co3d": ("o", "tab:blue", "CO3D"), "re10k": ("s", "tab:gray", "RealEstate10K"), "sevenscenes": ("^", "tab:green", "7-Scenes")}
    seen = set()
    for r in reps:
        if "smr" not in r["rows"] or "chained" not in r["rows"]: continue
        ds = r["dataset"].lower(); m, c, lab = mk.get(ds, ("x", "k", ds))
        K = r["keyframes"] or 1; nrev = r["n_revisit"] or r["rows"]["smr"].get("n_revisit_pairs", 0) or 0
        dens = nrev / max(1, K)                                     # GT revisit pairs per keyframe
        if ds == "sevenscenes":
            y = (r["rows"]["chained"]["ate_rmse"] - r["rows"]["smr"]["ate_rmse"]) / max(r["rows"]["chained"]["ate_rmse"], 1e-9) * 100; ylab = "ATE reduction (%)"
        else:
            y = r["rows"]["smr"]["auc30"] - r["rows"]["chained"]["auc30"]; ylab = "ΔAUC@30 (+SMR − raw)"
        ax.scatter(dens, y, marker=m, color=c, s=14, alpha=0.8, label=lab if lab not in seen else None); seen.add(lab)
    ax.axhline(0, color="0.7", lw=0.6); ax.set_xlabel("GT revisit pairs per keyframe (≥60 frames apart, <0.25 m, <30°)"); ax.set_ylabel("ΔAUC@30 (+SMR − raw)")
    ax.set_title("gain vs revisit density, one point per sequence"); ax.legend(frameon=False); save(fig, a.out)


# ----------------------------------------------------------------------------------------------- 3. settling
def cmd_settling(a):
    """The protocol of the tier-2 selective-detectability test (T11), followed through the settling iterations:
    formed grid states (valid), the same state displaced to an adjacent formed state (coherent), and a matched-norm
    displacement orthogonal to the coherent subspace (incoherent).  Residual = 1 - cos(g, W_hg h(g)) (the novelty
    signal); position error = decoded position vs the TRUE position x0."""
    from smr.memory import BlockScaffold, Novelty
    rng = np.random.default_rng(31)
    bs = BlockScaffold([2.4, 3.2, 4.0], torus_N=24, N_h=768, k=48, seed=0); spacing = 0.5
    bs.form_grid(extent=1.0, spacing=spacing); nov = Novelty().fit(bs); Q = bs.coherent_basis()
    res = {"valid": [], "incoherent": [], "coherent": []}; dist = {"valid": [], "incoherent": [], "coherent": []}
    for trial in range(a.trials):
        x0 = rng.integers(-1, 2, 3) * spacing; ph0 = bs.phases_of_pos(x0)
        ax_i = rng.integers(0, 3); d = np.zeros(3); d[ax_i] = spacing if x0[ax_i] < 1.0 else -spacing
        dphi_c = (2 * np.pi / bs.periods)[:, None] * d[None, :]
        z = rng.standard_normal(3 * bs.M); z -= Q @ (Q.T @ z); z *= a.sigma * np.linalg.norm(dphi_c) / np.linalg.norm(z)
        starts = {"valid": bs.encode_phases(ph0), "coherent": bs.encode_phases((ph0 + dphi_c) % (2 * np.pi)),
                  "incoherent": bs.encode_phases((ph0 + z.reshape(bs.M, 3)) % (2 * np.pi))}
        for name, g in starts.items():
            rr, dd = [nov.score(g)], [np.linalg.norm(bs.decode_pos(g) - x0)]
            for _ in range(a.iters):
                g = bs.settle(g, iters=1, damp=0.5); rr.append(nov.score(g)); dd.append(np.linalg.norm(bs.decode_pos(g) - x0))
            res[name].append(rr); dist[name].append(dd)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    cols = {"valid": "0.3", "incoherent": "tab:red", "coherent": "tab:blue"}; t = np.arange(a.iters + 1)
    for name in res:
        R = np.array(res[name]); D = np.array(dist[name])
        axes[0].plot(t, np.median(R, 0), color=cols[name], lw=1.4, label=name); axes[0].fill_between(t, np.percentile(R, 25, 0), np.percentile(R, 75, 0), color=cols[name], alpha=0.15, lw=0)
        axes[1].plot(t, np.median(D, 0), color=cols[name], lw=1.4, label=name); axes[1].fill_between(t, np.percentile(D, 25, 0), np.percentile(D, 75, 0), color=cols[name], alpha=0.15, lw=0)
    axes[0].set_yscale("log"); axes[0].set_xlabel("settling iteration"); axes[0].set_ylabel("novelty $1-\\cos(g, W_{hg}h(g))$")
    axes[0].set_title("detection: incoherent states light up the mismatch\nsignal; valid and coherent states sit at the floor")
    axes[1].set_xlabel("settling iteration"); axes[1].set_ylabel("decoded position error (m)")
    axes[1].set_title("correction: the loop only partly repairs incoherence and\ncannot see a coherent shift; both need a landmark re-anchor")
    axes[1].legend(frameon=False); save(fig, a.out)
    print({k: (round(float(np.median(np.array(res[k])[:, 0])), 4), round(float(np.median(np.array(res[k])[:, -1])), 4), round(float(np.median(np.array(dist[k])[:, -1])), 3)) for k in res})


# ----------------------------------------------------------------------------------------------- 4. funnel
def cmd_funnel(a):
    reps = load_reports(a.reports); ests = {pathlib.Path(f).stem: f for f in glob.glob(a.est)} if a.est else {}
    stages = dict(windows=0, with_gt_revisit=0, cand_lists=0, recall5=0, sites_verified=0, accepted=0, accepted_correct=0, rejected=0)
    cos_true, cos_false = [], []
    for r in reps:
        ev = (r["events"] or {}).get("smr") or []
        est_file = next((f for s, f in ests.items() if r["scene"] and r["scene"] in s), None) or next((f for s, f in ests.items() if s in r["name"] or r["name"] in s), None)
        gt = chunks = None
        if est_file:
            gt, _, chunks, _ = load_est(est_file)
        for e in ev:
            if not isinstance(e, dict) or "chunk" not in e: continue
            stages["windows"] += 1; k = int(e["chunk"])
            cands = e.get("candidates") or []; sites = e.get("sites") or []; loop = e.get("loop") or {}
            if cands: stages["cand_lists"] += 1
            correct_view = None
            if gt is not None and chunks is not None and k < len(chunks):
                cur = chunks[k]; earlier = np.concatenate([chunks[j] for j in range(max(0, k - 1))]) if k > 1 else np.array([], int)
                if len(earlier):
                    dpos = np.linalg.norm(gt[cur][:, None, :3, 3] - gt[earlier][None, :, :3, 3], axis=-1)
                    Rrel = np.einsum("iab,jac->ijbc", gt[cur][:, :3, :3], gt[earlier][:, :3, :3]); ang = np.degrees(np.arccos(np.clip((np.trace(Rrel, axis1=2, axis2=3) - 1) / 2, -1, 1)))
                    good = (dpos < a.dpos) & (ang < a.dang)
                    if good.any():
                        stages["with_gt_revisit"] += 1; correct_view = set(int(v) for v in earlier[good.any(0)])
                        if any(int(c["view"]) in correct_view for c in cands[:5]): stages["recall5"] += 1
                    for c in cands:
                        (cos_true if int(c["view"]) in (correct_view or set()) else cos_false).append(float(c.get("cos", 0)))
            if any(s.get("ok") for s in sites): stages["sites_verified"] += 1
            if loop.get("accepted"):
                stages["accepted"] += 1
                if correct_view is not None:
                    anc = next((int(s["view"]) for s in sites if s.get("ok") and "view" in s), None)
                    if anc is not None and anc in correct_view: stages["accepted_correct"] += 1
            elif loop: stages["rejected"] += 1
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw=dict(width_ratios=[1.4, 1]))
    keys = ["windows", "with_gt_revisit", "recall5", "sites_verified", "accepted", "accepted_correct"]
    labels = ["windows", "with a GT revisit", "correct view in top-5", "≥1 site verified", "closure accepted", "accepted & GT-correct"]
    vals = [stages[k] for k in keys]
    axes[0].barh(range(len(keys))[::-1], vals, color=["0.6", "0.5", "tab:blue", "tab:blue", "tab:green", "tab:green"], height=0.7)
    axes[0].set_yticks(range(len(keys))[::-1]); axes[0].set_yticklabels(labels)
    for i, v in enumerate(vals): axes[0].text(v + max(vals) * 0.01, len(keys) - 1 - i, str(v), va="center", fontsize=7)
    axes[0].set_xlabel("count over all windows"); axes[0].set_title(f"appearance proposes, geometry decides ({len(reps)} sequences)")
    if cos_true or cos_false:
        bins = np.linspace(0, 1, 26)
        axes[1].hist(cos_false, bins=bins, density=True, alpha=0.5, color="0.6", label=f"unrelated ({len(cos_false)})")
        axes[1].hist(cos_true, bins=bins, density=True, alpha=0.6, color="tab:green", label=f"true revisit ({len(cos_true)})")
        axes[1].set_xlabel("descriptor cosine of proposed candidates"); axes[1].set_ylabel("density"); axes[1].legend(frameon=False); axes[1].set_title("why appearance alone cannot decide")
    save(fig, a.out); print(stages)


# ----------------------------------------------------------------------------------------------- 5. flatbar
def cmd_flatbar(a):
    def load(root, ds, v):
        out = {}
        for f in glob.glob(f"{root}/{ds}/{v}/*.json"):
            d = json.load(open(f)); out[pathlib.Path(f).stem] = {r["method"]: r for r in d["rows"]}
        return out
    panels = []
    T7, F7, D7 = load(a.root, "7scenes", "default"), load(a.root, "7scenes", "index_flat"), load(a.root, "7scenes", "index_dynamics")
    if T7:
        ids = sorted(k for k in T7 if k in F7)
        panels.append(("7-Scenes seq-01: ATE (m), +SMR", [(T7[s]["smr"]["ate_rmse"], F7[s]["smr"]["ate_rmse"], D7.get(s, {}).get("smr", {}).get("ate_rmse")) for s in ids], ids))
    Tc, Fc, Dc = load(a.root, "co3d", "default"), load(a.root, "co3d", "index_flat"), load(a.root, "co3d", "index_dynamics")
    if Tc:
        ids = sorted(k for k in Tc if k in Fc)
        panels.append(("CO3D N=200: AUC@30, +SMR", [(Tc[s]["smr"]["auc30"], Fc[s]["smr"]["auc30"], Dc.get(s, {}).get("smr", {}).get("auc30")) for s in ids], [s.split("_")[0] for s in ids]))
    if a.split:
        rows = {}
        for f in glob.glob(f"{a.split}/*.json"):
            m = re.match(r"(\w+?)_seq(\d+)_\w+?_(template|flat)\.json", pathlib.Path(f).name)
            if not m: continue
            d = json.load(open(f)); rows.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = {x["method"]: x for x in d["rows"]}
        ids = sorted(k for k, v in rows.items() if "template" in v and "flat" in v)
        if ids: panels.append(("7-Scenes full test split (18 traj.): ATE (m), +SMR", [(rows[k]["template"]["smr"]["ate_rmse"], rows[k]["flat"]["smr"]["ate_rmse"], None) for k in ids], [f"{k[0][:4]}-{k[1]}" for k in ids]))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels) + 0.6, 2.7), squeeze=False); axes = axes[0]
    for ax, (title, vals, ids) in zip(axes, panels):
        t = np.array([v[0] for v in vals]); f = np.array([v[1] for v in vals]); x = np.arange(len(ids))
        ax.scatter(x - 0.12, t, color=COL["smr"], s=14, label="scaffold (template)", zorder=3); ax.scatter(x + 0.12, f, color=COL["flat"], s=14, marker="s", label="flat key–value", zorder=3)
        dyn = [v[2] for v in vals]
        if any(d is not None for d in dyn): ax.scatter(x, [d if d is not None else np.nan for d in dyn], color=COL["dyn"], s=10, marker="^", label="scaffold (settled)", zorder=3)
        for xi, ti, fi in zip(x, t, f): ax.plot([xi - 0.12, xi + 0.12], [ti, fi], color="0.75", lw=0.8, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(ids, rotation=70, fontsize=5.5); ax.set_title(title)
        better = (t < f).sum() if "ATE" in title else (t > f).sum(); ax.text(0.02, 0.96, f"scaffold better on {better}/{len(ids)}", transform=ax.transAxes, va="top", fontsize=7)
        if "AUC" in title: ax.set_ylim(min(f.min(), t.min()) - 3, 100)
    axes[0].legend(frameon=False, loc="lower right"); save(fig, a.out)


# ----------------------------------------------------------------------------------------------- 6. curves
def cmd_curves(a):
    reps = load_reports(a.reports); assert reps, "no reports"
    ds_list = sorted({r["dataset"].lower() for r in reps}); bbs = sorted({r["backbone"] for r in reps})
    fig, axes = plt.subplots(1, len(ds_list), figsize=(3.3 * len(ds_list), 2.6), squeeze=False); axes = axes[0]
    palette = plt.get_cmap("tab10")
    for ax, ds in zip(axes, ds_list):
        for bi, bb in enumerate(bbs):
            sel = [r for r in reps if r["dataset"].lower() == ds and r["backbone"] == bb and r["N"]]
            if not sel: continue
            Ns = sorted({r["N"] for r in sel})
            def mean_at(N, row, key="auc30"): 
                v = [r["rows"][row][key] for r in sel if r["N"] == N and row in r["rows"]]; return np.mean(v) if v else np.nan
            raw = [mean_at(N, "chained") for N in Ns]; smr = [mean_at(N, "smr") for N in Ns]; pgo = [mean_at(N, "smr_pgo") for N in Ns]
            ceil = [np.mean([r["ceiling"]["auc30"] for r in sel if r["N"] == N and r["ceiling"].get("auc30") is not None]) if any(r["ceiling"].get("auc30") is not None for r in sel if r["N"] == N) else np.nan for N in Ns]
            c = palette(bi)
            ax.plot(Ns, raw, color=c, ls=":", lw=1, marker="o", ms=2.5); ax.plot(Ns, smr, color=c, ls="-", lw=1.4, marker="o", ms=2.5, label=bb)
            if not all(np.isnan(pgo)): ax.plot(Ns, pgo, color=c, ls="--", lw=1, marker="o", ms=2)
            if not all(np.isnan(ceil)): ax.plot(Ns, ceil, color=c, ls="none", marker="*", ms=7, mfc="none")
        ax.set_xscale("log"); ax.set_xticks([10, 50, 100, 200]); ax.set_xticklabels(["10", "50", "100", "200"]); ax.set_xlabel("N frames"); ax.set_ylabel("AUC@30"); ax.set_title(ds)
    from matplotlib.lines import Line2D
    axes[-1].legend(handles=[Line2D([], [], color="k", ls=":", label="raw (chained windows)"), Line2D([], [], color="k", ls="-", label="+SMR"), Line2D([], [], color="k", ls="--", label="+SMR+PGO"), Line2D([], [], color="k", ls="none", marker="*", mfc="none", ms=7, label="native single pass (where it fits)")] + axes[0].get_legend_handles_labels()[0], frameon=False, fontsize=6, loc="lower left")
    save(fig, a.out)


# ----------------------------------------------------------------------------------------------- 7. timeline
def cmd_timeline(a):
    from smr.eval.trajectory import align_to_gt
    gt, rows, chunks, E = load_est(a.est); K = len(gt)
    G = np.load(str(E["gt"]), allow_pickle=True); key = np.asarray(E["key"], int)
    starts = []
    if "sessions" in G.files:
        sess = np.asarray(G["sessions"]); sess_k = sess[key]; starts = [i for i in range(1, K) if sess_k[i] != sess_k[i - 1]]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    for r, lab, col, ls in (("chained", a.raw_label, COL["raw"], "-"), ("smr", "+SMR (external bank persists)", COL["smr"], "-"), ("smr_pgo", "+SMR+PGO", COL["pgo"], "--")):
        if r not in rows: continue
        A, _ = align_to_gt(rows[r], gt); e = np.linalg.norm(A[:, :3, 3] - gt[:, :3, 3], axis=1)
        ax.plot(e, color=col, ls=ls, lw=1.0, label=f"{lab}: ATE {np.sqrt((e**2).mean()):.3f} m")
    for s in starts: ax.axvline(s, color="k", lw=1.0, ls="-.", alpha=0.8)
    if starts: ax.text(starts[0] + 1, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "session boundary:\nstreaming state reset", fontsize=6.5, va="bottom")
    ax.set_xlabel("keyframe"); ax.set_ylabel("position error (m)"); ax.set_title(a.title or pathlib.Path(a.est).stem); ax.legend(frameon=False, loc="upper right"); save(fig, a.out)


# ----------------------------------------------------------------------------------------------- 8. nvshist
def cmd_nvshist(a):
    jl = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}
    r1, r4 = jl(f"{a.nvs}/gso_reads1.jsonl"), jl(f"{a.nvs}/gso_reads{a.reads}.jsonl")
    ids = [k for k in r1 if k in r4 and "psnr" in r1[k] and "psnr" in r4[k]]; d = np.array([r4[k]["psnr"] - r1[k]["psnr"] for k in ids])
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.hist(d, bins=np.linspace(-2, 3, 51), color="tab:blue", alpha=0.75)
    ax.axvline(0, color="k", lw=0.8); ax.axvline(d.mean(), color="tab:red", lw=1.2, label=f"mean {d.mean():+.2f} dB; {(d>0).mean()*100:.0f}% improved")
    for mark in a.mark or []:
        hit = [k for k in ids if mark.lower() in k.lower()]
        if hit:
            v = r4[hit[0]]["psnr"] - r1[hit[0]]["psnr"]; ax.axvline(v, color="tab:green", lw=1.0, ls="--"); ax.text(v, ax.get_ylim()[1] * 0.9, f" {mark[:14]} {v:+.2f}", rotation=90, va="top", fontsize=6)
    ax.set_xlabel(f"ΔPSNR per object: {a.reads} orderings − single pass (dB)"); ax.set_ylabel("objects"); ax.set_title(f"GSO, {len(ids)} objects, {pathlib.Path(a.nvs).name} head"); ax.legend(frameon=False, fontsize=6.5)
    save(fig, a.out)


# ----------------------------------------------------------------------------------------------- 9. gpumem
def cmd_gpumem(a):
    reps = load_reports(a.reports); assert reps, "no reports"
    bbs = sorted({r["backbone"] for r in reps}); fig, ax = plt.subplots(figsize=(3.6, 2.6)); palette = plt.get_cmap("tab10")
    for bi, bb in enumerate(bbs):
        sel = [r for r in reps if r["backbone"] == bb and r["N"]]; Ns = sorted({r["N"] for r in sel})
        win = [np.mean([r["rows"]["smr"]["peak_mem_gb"] for r in sel if r["N"] == N and "smr" in r["rows"] and r["rows"]["smr"].get("peak_mem_gb")]) if any(r["rows"].get("smr", {}).get("peak_mem_gb") for r in sel if r["N"] == N) else np.nan for N in Ns]
        nat = [np.mean([r["ceiling"]["peak_gb"] for r in sel if r["N"] == N and r["ceiling"].get("peak_gb")]) if any(r["ceiling"].get("peak_gb") for r in sel if r["N"] == N) else np.nan for N in Ns]
        c = palette(bi); ax.plot(Ns, win, color=c, ls="-", marker="o", ms=3, lw=1.3, label=f"{bb} (+SMR, W=32)")
        if not all(np.isnan(nat)): ax.plot(Ns, nat, color=c, ls=":", marker="x", ms=4, lw=1, label=f"{bb} native N-frame pass")
    ax.axhline(a.gpu, color="0.5", lw=0.8, ls="--"); ax.text(0.02, 0.97, f"{a.gpu:.0f} GB GPU", transform=ax.transAxes, va="top", fontsize=6.5, color="0.4")
    ax.set_xscale("log"); ax.set_xticks([10, 50, 100, 200]); ax.set_xticklabels(["10", "50", "100", "200"]); ax.set_xlabel("N frames"); ax.set_ylabel("peak GPU memory (GB)"); ax.legend(frameon=False, fontsize=6); save(fig, a.out)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("seams"); p.add_argument("--est", required=True); p.add_argument("--out", default="outputs/figures/mech_seams"); p.set_defaults(f=cmd_seams)
    p = sub.add_parser("revisits"); p.add_argument("--reports", required=True); p.add_argument("--out", default="outputs/figures/mech_revisits"); p.set_defaults(f=cmd_revisits)
    p = sub.add_parser("settling"); p.add_argument("--iters", type=int, default=15); p.add_argument("--trials", type=int, default=40); p.add_argument("--sigma", type=float, default=0.6); p.add_argument("--shift", type=float, default=0.5); p.add_argument("--out", default="outputs/figures/mech_settling"); p.set_defaults(f=cmd_settling)
    p = sub.add_parser("funnel"); p.add_argument("--reports", required=True); p.add_argument("--est", default=""); p.add_argument("--dpos", type=float, default=0.25); p.add_argument("--dang", type=float, default=30); p.add_argument("--out", default="outputs/figures/mech_funnel"); p.set_defaults(f=cmd_funnel)
    p = sub.add_parser("flatbar"); p.add_argument("--root", default="outputs/ablate"); p.add_argument("--split", default=""); p.add_argument("--out", default="outputs/figures/mech_flatbar"); p.set_defaults(f=cmd_flatbar)
    p = sub.add_parser("curves"); p.add_argument("--reports", required=True); p.add_argument("--out", default="outputs/figures/mech_curves"); p.set_defaults(f=cmd_curves)
    p = sub.add_parser("timeline"); p.add_argument("--est", required=True); p.add_argument("--raw-label", default="native streaming state (reset per session)"); p.add_argument("--title", default=""); p.add_argument("--out", default="outputs/figures/mech_timeline"); p.set_defaults(f=cmd_timeline)
    p = sub.add_parser("nvshist"); p.add_argument("--nvs", default="outputs/nvs/vggt"); p.add_argument("--reads", type=int, default=4); p.add_argument("--mark", nargs="*", default=[]); p.add_argument("--out", default="outputs/figures/mech_nvshist"); p.set_defaults(f=cmd_nvshist)
    p = sub.add_parser("gpumem"); p.add_argument("--reports", required=True); p.add_argument("--gpu", type=float, default=46); p.add_argument("--out", default="outputs/figures/mech_gpumem"); p.set_defaults(f=cmd_gpumem)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
