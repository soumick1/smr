#!/usr/bin/env python3
"""Tier 1 -- dynamics ladder T1-T8 with figures and GIFs.

Every criterion mirrors the project plan; --fast shrinks sweeps for laptops,
full settings are the server defaults.
"""
import argparse
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt  # noqa: E402

from smr.dynamics import Ring, Torus2D, ScaffoldState, kernels as K  # noqa: E402
from smr.utils import Report, project_root, rng  # noqa: E402
from smr.utils.geometry import wrap_pi, euler_zyx_to_R, make_T  # noqa: E402
from smr.viz import TEAL, PURPLE, CORAL, GRAY, savefig, write_gif, fig_to_frame  # noqa: E402

OUT = project_root() / "outputs"


def t1_bump(rep, fast):
    r = Ring(seed=1)
    r.settle(600 if fast else 1200)
    w_meas, w_pred = r.width(), r.bump_width
    rep.check("T1 ring width vs closed form", round(w_meas, 3),
              abs(w_meas - w_pred) / w_pred < 0.10, "|dw|/w < 10%")
    # perturbation recovery
    u0 = r.u.copy()
    r.u *= (1 + 0.15 * rng(7).standard_normal(r.N))
    r.settle(300)
    prof_corr = np.corrcoef(np.sort(u0), np.sort(r.u))[0, 1]
    rep.check("T1 profile recovery after 15% pert", round(prof_corr, 4),
              prof_corr > 0.99, "sorted-profile corr > .99")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot(r.angles, r.u, color=TEAL, lw=1.6, label="settled $u(\\theta)$")
    th = r.decode()
    for s in (-1, 1):
        ax.axvline((th + s * w_pred / 2) % (2 * np.pi), color=CORAL, ls="--",
                   lw=1.2, label="closed-form edge" if s < 0 else None)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("$\\theta$ (rad)"); ax.set_ylabel("$u$")
    ax.set_title(f"T1: width {w_meas:.2f} vs closed form {w_pred:.2f}")
    ax.legend(loc="upper right")
    savefig(fig, OUT / "figures" / "T1_bump_profile.png")
    return r


def t2_gain(rep, fast):
    r = Ring(seed=2)
    slope = r.calibrate()
    vmax = r.measure_vmax()
    rates = np.linspace(0.02, 0.8 * vmax, 6 if fast else 11)
    realized = []
    for rr in rates:
        r.settle(250)
        realized.append(r._track(rr, 30 if fast else 50))
    realized = np.array(realized)
    dev = np.abs(realized - rates)
    ok = np.all((dev < 0.05 * rates) | (dev < 0.004))
    err = float(np.max(dev / rates))
    rep.check("T2 gain linearity (band)", round(err, 3), bool(ok),
              "per-rate: rel < 5% or abs < 0.004 rad/t")
    rep.check("T2 omega_max measured", round(vmax, 3), vmax >= 0.10,
              "omega_max >= 0.10 rad/t")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(rates, rates, color=GRAY, ls=":", label="identity")
    ax.plot(rates, realized, "o-", color=PURPLE, label="realized")
    ax.axvline(vmax, color=CORAL, ls="--", label=f"$\\omega_{{max}}$={vmax:.2f}")
    ax.set_xlabel("commanded rate (rad/t)"); ax.set_ylabel("realized")
    ax.set_title(f"T2: calibrated integration (raw slope {slope:.3f})")
    ax.legend()
    savefig(fig, OUT / "figures" / "T2_gain_vmax.png")
    return r, vmax


def t3_place(rep, fast, r):
    g = rng(11)
    errs = []
    for _ in range(6 if fast else 20):
        tgt = g.uniform(0, 2 * np.pi)
        errs.append(abs(wrap_pi(r.place(tgt) - tgt)))
    rep.check("T3 placement max err (rad)", round(float(np.max(errs)), 4),
              np.max(errs) < 0.02, "max err < 0.02 rad")


def t5_torus_path(rep, fast):
    t = Torus2D(seed=5)
    t.calibrate()
    t.place((3.0, 3.0))
    L, rate = 1.6, 0.08
    seq = [(rate, 0), (0, rate), (-rate, 0), (0, -rate)]
    xs, ys = [], []
    n = int(L / rate / t.dt)
    for vx, vy in seq:
        for _ in range(n):
            t.step((vx, vy))
            px, py = t.decode()
            xs.append(px); ys.append(py)
    err = np.hypot(wrap_pi(xs[-1] - 3.0), wrap_pi(ys[-1] - 3.0))
    rep.check("T5 closed-square return err (rad)", round(float(err), 3),
              err < 0.15, "return err < 0.15 rad")
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.plot(np.unwrap(xs), np.unwrap(ys), color=TEAL, lw=1.4)
    ax.plot(np.unwrap(xs)[0], np.unwrap(ys)[0], "o", color=CORAL, label="start/end")
    ax.set_xlabel("$\\phi_x$"); ax.set_ylabel("$\\phi_y$"); ax.set_title(
        f"T5: closed square, return err {err:.3f} rad")
    ax.legend()
    savefig(fig, OUT / "figures" / "T5_torus_path.png")


def t6_gate(rep, fast):
    ss = ScaffoldState(periods=[2.4, 3.2], ring_N=128, torus_N=32, seed=6)
    ss.calibrate()
    def run(gate, yaw_spin=False):
        T0 = make_T(np.eye(3), np.zeros(3))
        ss.place_pose(T0)
        if yaw_spin:
            for _ in range(500 if fast else 1200):
                ss.step_ego((0.10, 0, 0), (0, 0, 0), gate=gate)
            return np.linalg.norm(ss.decode_position())
        # path A: +x then turn 90deg then +x(ego) ; path B reversed order
        def leg_fwd():
            for _ in range(220):
                ss.step_ego((0, 0, 0), (0.05, 0, 0), gate=gate)
        def leg_turn():
            for _ in range(int(np.pi / 2 / 0.10 / ss.dt)):
                ss.step_ego((0.10, 0, 0), (0, 0, 0), gate=gate)
        ss.place_pose(T0); leg_fwd(); leg_turn(); leg_fwd()
        xa = ss.decode_position()
        ss.place_pose(T0); leg_turn(); leg_fwd()
        yawR = euler_zyx_to_R(-np.pi / 2, 0, 0)   # pre-rotated equivalent leg
        for _ in range(220):
            ss.step_ego((0, 0, 0), yawR @ np.array([0.05, 0, 0]), gate=gate)
        xb = ss.decode_position()
        return np.linalg.norm(xa - xb)
    hol_on = run(True, yaw_spin=True)
    hol_off = run(False, yaw_spin=True)
    pin_on = run(True)
    pin_off = run(False)
    rep.check("T6 holonomy drift, gated (units)", round(float(hol_on), 3),
              hol_on < 0.08, "pure spin moves position < 0.08")
    rep.check("T6 path-inv gap gated vs ungated",
              (round(float(pin_on), 3), round(float(pin_off), 3)),
              pin_on < 0.15 and pin_off > 3 * max(pin_on, 1e-3),
              "gated < 0.15 and ungated >= 3x gated")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.bar(["holonomy\n(gated)", "holonomy\n(ungated)", "path-inv\n(gated)",
            "path-inv\n(ungated)"], [hol_on, hol_off, pin_on, pin_off],
           color=[TEAL, GRAY, PURPLE, CORAL])
    ax.set_ylabel("endpoint discrepancy")
    ax.set_title("T6: HD gating is what buys path invariance")
    savefig(fig, OUT / "figures" / "T6_gate.png")


def t7_residue(rep, fast):
    ss = ScaffoldState(periods=[2.4, 3.2, 4.0], ring_N=128, torus_N=32, seed=7)
    ss.calibrate()
    g = rng(17)
    errs = []
    for _ in range(6 if fast else 16):
        x = g.uniform(-1.1, 1.1, 3)
        ss.place_pose(make_T(np.eye(3), x))
        errs.append(np.linalg.norm(ss.decode_position() - x))
    rep.check("T7 residue decode max err", round(float(np.max(errs)), 4),
              np.max(errs) < 0.05, "max err < 0.05 units")
    # CRT fragility on the toy integer system p=(3,4,5)
    p = np.array([3, 4, 5]); R = np.prod(p)
    x_true = 23
    resid = x_true % p
    cands = np.arange(R)
    def decode(res):
        d = np.abs(((cands[:, None] - res[None, :]) + p // 2) % p - p // 2)
        return int(cands[np.argmin((d ** 2).sum(1))])
    ok = decode(resid)
    bad = resid.copy(); bad[1] = (bad[1] + 2) % p[1]     # single-module error
    jump = abs(decode(bad) - x_true)
    rep.check("T7 CRT single-module corruption jump", int(jump),
              jump > R // p[1] // 2, "relocation O(R/p_m), not O(1)")
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.bar(["clean", "1-module\ncorrupt"], [abs(ok - x_true), jump],
           color=[TEAL, CORAL])
    ax.set_ylabel("|decoded - true|")
    ax.set_title("T7: exact-CRT fragility (toy $p{=}3,4,5$)")
    savefig(fig, OUT / "figures" / "T7_crt_fragility.png")


def t8_rt(rep, fast, vmax):
    r = Ring(seed=8)
    r.calibrate()
    deltas = np.linspace(0.3, np.pi * 0.95, 5 if fast else 9)
    times = []
    for d in deltas:
        r.place(0.0)
        t, th = 0.0, 0.0
        while abs(wrap_pi(r.decode() - d)) > 0.03 and t < 400:
            e = wrap_pi(d - r.decode())
            r.step(np.clip(e / r.dt, -0.8 * vmax, 0.8 * vmax))
            t += r.dt
        times.append(t)
    A = np.vstack([deltas, np.ones_like(deltas)]).T
    coef, res, *_ = np.linalg.lstsq(A, np.array(times), rcond=None)
    ss_tot = np.sum((times - np.mean(times)) ** 2)
    r2 = 1 - (res[0] / ss_tot if len(res) else 0.0)
    rep.check("T8 RT-vs-angle linearity R^2", round(float(r2), 4), r2 > 0.98,
              "R^2 > 0.98")
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(np.degrees(deltas), times, "o", color=PURPLE, label="measured")
    ax.plot(np.degrees(deltas), A @ coef, color=TEAL,
            label=f"fit: T = {coef[0]:.2f}$\\Delta$ + {coef[1]:.2f}  ($R^2$={r2:.3f})")
    ax.set_xlabel("rotation magnitude (deg)"); ax.set_ylabel("time to arrive (t)")
    ax.set_title("T8: mental-rotation reaction-time law")
    ax.legend()
    savefig(fig, OUT / "figures" / "T8_rt_law.png")


def gifs(fast):
    # ring integration gif
    r = Ring(seed=21); r.calibrate(); r.place(0.6)
    frames = []
    for i in range(90 if fast else 160):
        for _ in range(6):
            r.step(0.12)
        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        ax.plot(r.angles, r.u, color=TEAL, lw=1.5)
        ax.axvline(r.decode(), color=CORAL, ls="--", lw=1.2)
        ax.set_ylim(-3.2, 1.1); ax.set_xlim(0, 2 * np.pi)
        ax.set_title(f"HD ring integrating $\\omega$=0.12 rad/t   "
                     f"$\\hat\\theta$={np.degrees(r.decode()):5.1f}$^\\circ$")
        ax.set_xlabel("$\\theta$"); ax.set_ylabel("$u$")
        frames.append(fig_to_frame(fig))
    write_gif(frames, OUT / "gifs" / "ring_integration.gif")
    # torus bump gif
    t = Torus2D(seed=22); t.calibrate(); t.place((1.0, 1.0))
    frames = []
    for i in range(70 if fast else 130):
        for _ in range(6):
            t.step((0.10, 0.06))
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        ax.imshow(t.f(t.u).T, origin="lower", cmap="viridis",
                  extent=[0, 2 * np.pi, 0, 2 * np.pi])
        px, py = t.decode()
        ax.plot(px, py, "o", color=CORAL, ms=5)
        ax.set_title("grid-module torus bump")
        ax.set_xlabel("$\\phi_x$"); ax.set_ylabel("$\\phi_y$")
        frames.append(fig_to_frame(fig))
    write_gif(frames, OUT / "gifs" / "torus_bump.gif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    rep = Report("Tier 1 -- dynamics (T1-T8)")
    print(rep.title)
    r1 = t1_bump(rep, a.fast)
    _, vmax = t2_gain(rep, a.fast)
    t3_place(rep, a.fast, r1)
    t5_torus_path(rep, a.fast)
    t6_gate(rep, a.fast)
    t7_residue(rep, a.fast)
    t8_rt(rep, a.fast, vmax)
    gifs(a.fast)
    rep.save(OUT / "reports" / "tier1.json")
    sys.exit(0 if rep.all_passed else 1)


if __name__ == "__main__":
    main()
