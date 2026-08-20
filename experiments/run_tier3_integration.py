#!/usr/bin/env python3
"""Tier 3 -- integration I1/I2/I3/I5 on the synthetic room, plus the
rotation-sweep GIF (scaffold activity + recalled-surfel render, live)."""
import argparse, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt                     # noqa: E402
import matplotlib.gridspec as gridspec              # noqa: E402
from smr.backbones import get_backbone              # noqa: E402
from smr.dynamics import ScaffoldState              # noqa: E402
from smr.pipeline import bind, rotate               # noqa: E402
from smr.scene import camera_ring, render           # noqa: E402
from smr.utils import Report, project_root, load_config, rng  # noqa: E402
from smr.utils.geometry import (pose_errors, make_T, euler_zyx_to_R,
                                wrap_pi)            # noqa: E402
from smr.viz import (TEAL, PURPLE, CORAL, GRAY, savefig, write_gif,
                     fig_to_frame, to_uint8)        # noqa: E402

OUT = project_root() / "outputs"


def decode_pos_from_phases(ph, periods):
    order = np.argsort(-np.asarray(periods))
    x = np.zeros(3)
    for j_i, j in enumerate(order):
        lam, frac = periods[j], ph[j] / (2 * np.pi)
        if j_i == 0:
            x = ((frac + 0.5) % 1.0 - 0.5) * lam
        else:
            x = x + lam * (((frac - x / lam) + 0.5) % 1.0 - 0.5)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--config", default="compact")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rep = Report("Tier 3 -- integration (I1-I5)"); print(rep.title)

    bb = get_backbone("synthetic", seed=0)
    out = bb.infer(n_views=cfg["views"], radius=cfg["radius"])
    cam = bb.cam
    periods = cfg["periods"]

    ss = ScaffoldState(periods=periods, ring_N=cfg["ring_N"],
                       torus_N=cfg["torus_N"], seed=0)
    cal = ss.calibrate(verbose=True)

    # ---------------- I1: placement -> pose decode
    rot_errs, pos_errs = [], []
    for i in range(out.poses.shape[0]):
        ss.place_pose(out.poses[i])
        T_est = make_T(ss.decode_R(), ss.decode_position())
        re_, pe_ = pose_errors(T_est, out.poses[i])
        rot_errs.append(re_); pos_errs.append(pe_)
    rep.check("I1 place->decode rot err (rad, max)",
              round(float(np.max(rot_errs)), 4), np.max(rot_errs) < 0.02,
              "max rot err < 0.02 rad (~1.1 deg)")
    rep.check("I1 place->decode pos err (max)",
              round(float(np.max(pos_errs)), 4), np.max(pos_errs) < 0.05,
              "max pos err < 0.05 (~1% scene scale)")

    # ---------------- bind the scene
    bound = bind(out, ss, cam, formation_extent=cfg["formation"]["extent"],
                 formation_spacing=cfg["formation"]["spacing"],
                 torus_N=cfg["torus_N"], N_h=cfg["N_h"], k=cfg["k"], seed=0)

    # ---------------- I2: drive to a held pose
    ss.place_pose(out.poses[0])
    n_steps, _ = ss.drive_to(out.poses[3])
    T_est = make_T(ss.decode_R(), ss.decode_position())
    re_, pe_ = pose_errors(T_est, out.poses[3])
    rep.check("I2 drive_to pose err (rot, pos)",
              (round(re_, 4), round(pe_, 4)), re_ < 0.05 and pe_ < 0.08,
              f"rot<0.05 rad, pos<0.08 after {n_steps} steps")

    # ---------------- I3: relocalisation from corrupted descriptor
    g0 = rng(77); correct = 0
    Hs = [bound.store.content[j]["h"] for j in range(len(bound.store.states))]
    for i in range(out.poses.shape[0]):
        st = out.descriptor(i) + 0.2 * g0.standard_normal(448)
        st /= np.linalg.norm(st)
        hh = bound.mem.cue_h(st)
        jstar = int(np.argmax([hh @ hj for hj in Hs]))
        correct += int(jstar == i)
    rep.check("I3 relocalisation top-1 (of {})".format(out.poses.shape[0]),
              correct, correct >= out.poses.shape[0] - 1,
              "corrupted-cue view id correct (allow 1 miss)")

    # ---------------- I5: novel-pose readout vs ground truth
    T_q = camera_ring(16, radius=cfg["radius"], seed=0)[3]   # between views 1,2
    ss.place_pose(out.poses[1])
    res = rotate(bound, T_q, cam, k_recall=2, settle_fiber=False)
    rgb_gt, dep_gt, msk_gt = render(bb.points, bb.colors, T_q, cam)
    both = res["mask"] & msk_gt
    med_dz = float(np.median(np.abs(res["depth"][both] - dep_gt[both])))
    rep.check("I5 seen-surface depth med|dz|", round(med_dz, 4),
              med_dz < 0.08 and both.mean() > 0.15,
              "median depth err < 0.08 on >15% joint coverage")
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    axs[0].imshow(np.clip(res["rgb"], 0, 1)); axs[0].set_title("recalled splat @ novel pose")
    axs[1].imshow(np.clip(rgb_gt, 0, 1));     axs[1].set_title("ground truth")
    err_img = np.where(both, np.abs(res["depth"] - dep_gt), np.nan)
    im = axs[2].imshow(err_img, cmap="magma", vmax=0.2)
    axs[2].set_title(f"|depth err|, med {med_dz:.3f}")
    for ax in axs: ax.axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    savefig(fig, OUT / "figures" / "I5_novel_pose.png")

    # ---------------- rotation sweep GIF (the concept, live)
    snaps = []
    def snap(s):
        snaps.append(dict(yaw_u=s.rings["yaw"].u.copy(),
                          tor=s.modules[0].torus.f(s.modules[0].torus.u).copy(),
                          xi=s.state().copy()))
    ss.place_pose(out.poses[0]); snap(ss)
    ss.drive_to(out.poses[3], snapshot_fn=snap,
                snapshot_every=6 if a.fast else 4)
    frames = []
    stride = max(1, len(snaps) // (55 if a.fast else 90))
    for sn in snaps[::stride]:
        xi = sn["xi"]
        R = euler_zyx_to_R(*xi[:3])
        pos = decode_pos_from_phases(xi[3:].reshape(len(periods), 3), periods)
        T_f = make_T(R, pos)
        ids = bound.store.nearest(xi, k=2)
        P, C = [], []
        from smr.scene import transform, splat
        for j in ids:
            pay = bound.store.content[j]
            P.append(transform(pay["pts"], pay["T"], T_f)); C.append(pay["cols"])
        rgb, dep, msk = splat(np.concatenate(P), np.concatenate(C), T_f, cam)
        fig = plt.figure(figsize=(9.2, 3.1))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.25)
        ax0 = fig.add_subplot(gs[0]); ax0.imshow(np.clip(rgb, 0, 1)); ax0.axis("off")
        ax0.set_title(f"recalled render  yaw {np.degrees(xi[0]):6.1f}$^\\circ$")
        ax1 = fig.add_subplot(gs[1])
        angles = np.linspace(0, 2 * np.pi, len(sn["yaw_u"]), endpoint=False)
        ax1.plot(angles, sn["yaw_u"], color=TEAL, lw=1.4)
        ax1.axvline(xi[0] % (2 * np.pi), color=CORAL, ls="--", lw=1.1)
        ax1.set_ylim(-3.5, 1.2); ax1.set_title("HD yaw ring $u(\\theta)$")
        ax1.set_xlabel("$\\theta$")
        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(sn["tor"].T, origin="lower", cmap="viridis")
        ax2.set_title("grid module 0 (torus)"); ax2.axis("off")
        frames.append(fig_to_frame(fig))
    write_gif(frames, OUT / "gifs" / "rotation_sweep.gif", fps=12)

    rep.save(OUT / "reports" / "tier3.json")
    sys.exit(0 if rep.all_passed else 1)


if __name__ == "__main__":
    main()
