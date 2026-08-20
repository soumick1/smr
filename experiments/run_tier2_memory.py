#!/usr/bin/env python3
"""Tier 2 -- memory ladder T9-T14 with figures."""
import argparse, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt  # noqa: E402
from smr.memory import (ToyVectorHash, nh_star, pseudoinverse_recall_error,
                        hopfield_recall_error, BlockScaffold, RLSMemory,
                        Novelty)  # noqa: E402
from smr.memory.toy_vectorhash import all_states, encode  # noqa: E402
from smr.utils import Report, project_root, rng  # noqa: E402
from smr.viz import TEAL, PURPLE, CORAL, GRAY, savefig  # noqa: E402

OUT = project_root() / "outputs"


def t9(rep, fast):
    fam = [[3, 4], [3, 4, 5], [3, 4, 5, 7]] + ([] if fast else [[3, 4, 5, 7, 11]])
    Ms = [len(p) for p in fam]
    ns = [nh_star(p, hi=400 if fast else 800) for p in fam]
    A = np.vstack([Ms, np.ones_like(Ms)]).T
    coef, res, *_ = np.linalg.lstsq(A, np.array(ns, float), rcond=None)
    sstot = np.sum((ns - np.mean(ns)) ** 2)
    r2 = 1 - (res[0] / sstot if len(res) else 0.0)
    rep.check("T9 N_h* linear in M (R^2)", round(float(r2), 3), r2 > 0.9,
              "linear fit R^2 > 0.9")
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.plot(Ms, ns, "o", color=PURPLE)
    ax.plot(Ms, A @ coef, color=TEAL, label=f"$R^2$={r2:.3f}")
    ax.set_xlabel("modules M"); ax.set_ylabel("$N_h^*$")
    ax.set_title("T9: scaffold size is linear in M (toy VH)")
    ax.legend()
    savefig(fig, OUT / "figures" / "T9_nhstar.png")


def t10(rep, fast):
    P = np.array([5, 10, 20, 40, 80, 120, 160]) if not fast else \
        np.array([5, 10, 20, 40, 80])
    e_pinv = pseudoinverse_recall_error(200, P)
    e_hop = hopfield_recall_error(200, P)
    graceful = np.all(np.diff(e_pinv) < 0.25) and e_pinv[0] < 0.05
    cliff = e_hop[-1] > 0.15 and e_hop[0] < 0.02
    rep.check("T10 pinv graceful / Hopfield cliff",
              (round(float(e_pinv[-1]), 3), round(float(e_hop[-1]), 3)),
              bool(graceful and cliff),
              "pinv smooth from ~0; Hopfield collapses past capacity")
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ax.plot(P, e_pinv, "o-", color=TEAL, label="scaffold + pinv content")
    ax.plot(P, e_hop, "s--", color=CORAL, label="Hopfield control")
    ax.axvline(0.138 * 200, color=GRAY, ls=":", label="0.138 N")
    ax.set_xlabel("stored items P"); ax.set_ylabel("recall error")
    ax.set_title("T10: memory continuum vs catastrophe")
    ax.legend()
    savefig(fig, OUT / "figures" / "T10_continuum.png")


def t11(rep, fast):
    """Selective DETECTABILITY (Thm. selective): coherent displacements map
    formed states to formed states -- reconstruction residual stays at the
    formed-state floor -- while matched-norm incoherent displacements leave
    the manifold and light up the mismatch signal.  (The loop is the
    detector; correction proper is the landmark clamp, which base-blindness
    makes mandatory anyway.)"""
    bs = BlockScaffold([2.4, 3.2, 4.0], torus_N=24, N_h=768, k=48, seed=0)
    spacing = 0.5
    bs.form_grid(extent=1.0, spacing=spacing)
    nov = Novelty().fit(bs)
    g0 = rng(31)
    Q = bs.coherent_basis()
    r_par, r_perp = [], []
    for _ in range(10 if fast else 24):
        x0 = (g0.integers(-1, 2, 3)) * spacing
        ph0 = bs.phases_of_pos(x0)
        ax_i = g0.integers(0, 3)
        d = np.zeros(3); d[ax_i] = spacing if x0[ax_i] < 1.0 else -spacing
        dphi_c = (2 * np.pi / bs.periods)[:, None] * d[None, :]
        z = g0.standard_normal(3 * bs.M)
        z -= Q @ (Q.T @ z)
        z *= np.linalg.norm(dphi_c) / np.linalg.norm(z)
        r_par.append(nov.score(bs.encode_phases((ph0 + dphi_c) % (2 * np.pi))))
        r_perp.append(nov.score(bs.encode_phases((ph0 + z.reshape(bs.M, 3))
                                                 % (2 * np.pi))))
    med_p, med_q = float(np.median(r_par)), float(np.median(r_perp))
    ratio = med_q / max(med_p, 1e-9)
    rep.check("T11 selective detectability (perp/par residual)",
              round(ratio, 1), ratio > 50.0 and med_p < 0.02,
              "coherent invisible (<0.02), incoherent flagged; >50x")
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.boxplot([r_par, r_perp], tick_labels=["coherent\n(along C)",
                                             "incoherent\n(perp C)"])
    ax.set_yscale("log")
    ax.set_ylabel("reconstruction residual $1-\\cos(g,\\hat g)$")
    ax.set_title(f"T11: detectability is selective ({ratio:.0f}x)")
    savefig(fig, OUT / "figures" / "T11_anisotropy.png")
    return bs


def t12(rep, fast, bs):
    g0 = rng(41)
    xs = [np.array([x, y, 0.0]) for x in (-1, 0, 1) for y in (-1, 0, 1)][:8]
    H = [bs.h_of(bs.encode_pos(x)) for x in xs]
    S = [g0.standard_normal(448) for _ in xs]
    S = [s / np.linalg.norm(s) for s in S]
    mem = RLSMemory(bs.N_h, 448)
    for h, s in zip(H, S):
        mem.write(h, s)
    sigmas = np.arange(0.0, 0.65, 0.05)
    errs = []
    for sg in sigmas:
        e = []
        for j, (h, s) in enumerate(zip(H, S)):
            st = s + sg * g0.standard_normal(448); st /= np.linalg.norm(st)
            hh = mem.cue_h(st)
            jstar = int(np.argmax([hh @ hj for hj in H]))
            e.append(np.linalg.norm(mem.read_s(H[jstar]) - s))
        errs.append(np.max(e))
    errs = np.array(errs)
    exact = errs < 1e-2
    sig_star = float(sigmas[np.argmax(~exact) - 1]) if not exact.all() \
        else float(sigmas[-1])
    rep.check("T12 exact-recall threshold sigma*", sig_star, sig_star >= 0.25,
              "recall EXACT (not degraded) up to sigma* >= 0.25")
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.plot(sigmas, errs, "o-", color=PURPLE)
    ax.set_xlabel("cue corruption $\\sigma$"); ax.set_ylabel("content recall error")
    ax.set_title("T12: recall quality is cue-independent up to threshold")
    savefig(fig, OUT / "figures" / "T12_cue.png")


def t13(rep, fast, bs):
    nov = Novelty().fit(bs)
    g0 = rng(51)
    s_fam = [nov.score(bs._formed_g[:, i]) for i in
             g0.integers(0, bs._formed_g.shape[1], 60)]
    s_nov = [nov.score(g0.standard_normal(bs.N_g)) for _ in range(60)]
    lab = np.r_[np.zeros(len(s_fam)), np.ones(len(s_nov))]
    sc = np.r_[s_fam, s_nov]
    order = np.argsort(sc)
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(sc))
    auroc = (ranks[lab == 1].mean() - (lab.sum() - 1) / 2) / (lab == 0).sum()
    rep.check("T13 novelty AUROC", round(float(auroc), 3), auroc > 0.95,
              "AUROC > 0.95")
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.hist(s_fam, bins=20, alpha=0.7, color=TEAL, label="familiar")
    ax.hist(s_nov, bins=20, alpha=0.7, color=CORAL, label="novel")
    ax.set_xlabel("novelty score"); ax.set_ylabel("count")
    ax.set_title(f"T13: novelty detection, AUROC {auroc:.3f}")
    ax.legend()
    savefig(fig, OUT / "figures" / "T13_novelty.png")


def t14(rep, fast):
    """Strong generalisation is over COMBINATIONS: a random 20% of joint
    states (which covers every per-module phase w.h.p.) locks in the
    unseen 80% as fixed points.  Negative control: a contiguous block that
    excludes some phases entirely cannot generalise to them -- those
    one-hot units never appear in the training span."""
    periods = [5, 7, 9]
    allc = list(all_states(periods))
    g0 = rng(61)

    def pinv_form(states):
        tv = ToyVectorHash(periods, N_h=420, seed=3)
        G = np.stack([encode(c, periods) for c in states], 1)
        H = np.stack([tv.h_of(G[:, i]) for i in range(G.shape[1])], 1)
        tv.W_hg = G @ np.linalg.pinv(H)
        return tv

    sel = [allc[i] for i in g0.choice(len(allc), int(0.2 * len(allc)),
                                      replace=False)]
    tv_r = pinv_form(sel)
    unseen_r = [c for c in allc if c not in sel]
    frac_r = float(np.mean([tv_r.is_fixed(c) for c in unseen_r]))
    contig = [c for c in allc if c[0] < 3]
    tv_c = pinv_form(contig)
    frac_c = float(np.mean([tv_c.is_fixed(c) for c in allc if c[0] >= 3]))
    rep.check("T14 unseen-combination fixed frac (random 20%)",
              round(frac_r, 3), frac_r > 0.9,
              "combinatorial generalisation > 0.9")
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.bar(["random 20% seen\n(unseen combos)",
            "contiguous seen\n(excluded phases)"],
           [frac_r, frac_c], color=[TEAL, CORAL])
    ax.set_ylim(0, 1.05); ax.set_ylabel("unseen fixed-point fraction")
    ax.set_title("T14: generalisation over combinations, not phases")
    savefig(fig, OUT / "figures" / "T14_stronggen.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    rep = Report("Tier 2 -- memory (T9-T14)"); print(rep.title)
    t9(rep, a.fast); t10(rep, a.fast)
    bs = t11(rep, a.fast)
    t12(rep, a.fast, bs); t13(rep, a.fast, bs); t14(rep, a.fast)
    rep.save(OUT / "reports" / "tier2.json")
    sys.exit(0 if rep.all_passed else 1)


if __name__ == "__main__":
    main()
