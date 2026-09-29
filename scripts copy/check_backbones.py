#!/usr/bin/env python3
"""Backbone integration report: does each backbone load, infer, and BIND?

This is the weights-level companion to tests/test_backbones.py (which pins
the conversion maths without a GPU).  For every requested backbone it walks
five stages and reports where it stops:

    import  -> the third-party package is importable
    load    -> weights load onto the device
    infer   -> a real scene produces a BackboneOutput
    schema  -> shapes, dtypes, rigid poses, scale gauge all correct
    bind    -> the scaffold binds it and recalls the right view

Usage
    python scripts/check_backbones.py --frames lab_photos/room/images_8
    python scripts/check_backbones.py --frames orbit_frames \
           --backbones vggt pi3 dust3r --max-views 4
    python scripts/check_backbones.py --frames orbit_frames --json report.json

Nothing here is fatal: a failing backbone prints its stage and error and the
run continues, so one report tells you the state of the whole matrix.
"""
import argparse, json, pathlib, sys, time, traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smr.backbones import get_backbone                       # noqa: E402
from smr.backbones.base import _REGISTRY                     # noqa: E402
from smr.dynamics import ScaffoldState                       # noqa: E402
from smr.pipeline import bind                                # noqa: E402
from smr.scene import Camera                                 # noqa: E402

STAGES = ("construct", "import", "weights", "load", "infer",
          "schema", "bind")


def check_schema(out, n):
    K_, H, W = out.rgb.shape[:3]
    assert K_ == n, f"expected {n} views, got {K_}"
    assert out.poses.shape == (K_, 4, 4), f"poses {out.poses.shape}"
    assert out.depth.shape == (K_, H, W), f"depth {out.depth.shape}"
    assert out.mask.shape == (K_, H, W), f"mask {out.mask.shape}"
    assert out.mask.any(), "empty confidence mask"
    assert np.isfinite(out.poses).all(), "non-finite poses"
    med = float(np.median(out.depth[out.mask]))
    assert abs(med - 1.0) < 1e-4, f"scale gauge broken: median depth {med}"
    for i in range(K_):
        R = out.poses[i, :3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-3), f"view {i} not rigid"
        assert abs(np.linalg.det(R) - 1) < 1e-3, f"view {i} det {np.linalg.det(R)}"
    return dict(K=K_, H=H, W=W, coverage=round(float(out.mask.mean()), 3),
                focal=round(float(out.intrinsics[0, 0]), 2))


def check_bind(out):
    cam = Camera(H=out.depth.shape[1], W=out.depth.shape[2],
                 f=float(out.intrinsics[0, 0]))
    ss = ScaffoldState(periods=[2.4, 3.2, 4.0], ring_N=128, torus_N=32,
                       seed=0, omega_max=0.16)
    ss.calibrate()
    bound = bind(out, ss, cam, formation_extent=1.2, formation_spacing=0.4,
                 torus_N=32, N_h=1024, k=64, seed=0)
    hits = 0
    for i in range(out.poses.shape[0]):
        ss.place_pose(out.poses[i])
        if bound.store.nearest(ss.state(), k=1)[0] == i:
            hits += 1
    return dict(stored=len(bound.store.content), self_recall=hits,
                of=out.poses.shape[0])


def run_one(name, paths, device, kwargs):
    rec = dict(backbone=name, stage="construct", ok=False, seconds=None,
               info={}, error=None)
    t0 = time.time()
    try:
        try:
            bb = get_backbone(name, device=device, **kwargs)
        except TypeError:                  # adapters that take no device
            bb = get_backbone(name, **kwargs)
        rec["stage"] = "import"
        pre = bb.preflight() if hasattr(bb, "preflight") else {}
        rec["info"]["modules"] = pre
        missing = [m for m, ok in pre.items() if not ok]
        if missing:
            raise ImportError(f"missing module(s): {missing}")
        rec["info"]["pose_convention"] = getattr(bb, "pose_convention", "?")

        rec["stage"] = "weights"
        if hasattr(bb, "resolve_weights"):
            w = bb.resolve_weights(getattr(bb, "_weights_override", None))
            rec["info"]["weights"] = w
            rec["info"]["weights_local"] = pathlib.Path(w).is_file()
        else:
            rec["info"]["weights"] = getattr(bb, "weights", None)

        rec["stage"] = "load"
        if hasattr(bb, "_load") and getattr(bb, "_model", None) is None:
            bb._load()

        rec["stage"] = "infer"
        out = bb.infer([str(p) for p in paths])
        rec["info"]["infer_s"] = round(time.time() - t0, 1)

        rec["stage"] = "schema"
        rec["info"].update(check_schema(out, len(paths)))

        rec["stage"] = "bind"
        rec["info"].update(check_bind(out))

        rec["ok"] = True
        rec["stage"] = "done"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-1200:]
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True,
                    help="directory of images from ONE scene")
    ap.add_argument("--backbones", nargs="*", default=None,
                    help="default: every registered backbone except synthetic")
    ap.add_argument("--max-views", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="outputs/reports/backbone_check.json")
    ap.add_argument("--verbose-errors", action="store_true")
    a = ap.parse_args()

    d = pathlib.Path(a.frames)
    paths = sorted(p for p in d.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert len(paths) >= 2, f"need >=2 images in {d}"
    if len(paths) > a.max_views:
        step = len(paths) / a.max_views
        paths = [paths[int(i * step)] for i in range(a.max_views)]

    names = a.backbones or [n for n in sorted(_REGISTRY) if n != "synthetic"]
    print(f"scene {d}  |  {len(paths)} views  |  device {a.device}")
    print(f"backbones: {', '.join(names)}\n")

    rows = []
    for n in names:
        print(f"--- {n} " + "-" * (58 - len(n)))
        rec = run_one(n, paths, a.device, {})
        rows.append(rec)
        if rec["ok"]:
            i = rec["info"]
            print(f"  PASS  {i['K']}x{i['H']}x{i['W']}  conv={i['pose_convention']}"
                  f"  f={i['focal']}  coverage={i['coverage']}"
                  f"  recall={i['self_recall']}/{i['of']}  {rec['seconds']}s")
        else:
            print(f"  FAIL at [{rec['stage']}]  {rec['error']}  ({rec['seconds']}s)")
            tb = rec.get("traceback", "")
            if a.verbose_errors:
                print(tb)
            elif tb:
                # the originating frame is what identifies a third-party
                # incompatibility; one line saves a whole extra run
                frames = [ln.strip() for ln in tb.splitlines()
                          if ln.strip().startswith("File ")]
                if frames:
                    print(f"         raised at: {frames[-1]}")

    print("\n" + "=" * 66)
    print(f"{'backbone':<14}{'stage':<10}{'conv':<6}{'recall':<9}{'time':>7}")
    print("-" * 66)
    for r in rows:
        i = r["info"]
        rec = (f"{i.get('self_recall', '-')}/{i.get('of', '-')}"
               if r["ok"] else "-")
        print(f"{r['backbone']:<14}"
              f"{('OK' if r['ok'] else str(r['stage'])):<10}"
              f"{str(i.get('pose_convention', '-')):<6}{rec:<9}"
              f"{r['seconds']:>6}s")
    ok = sum(r["ok"] for r in rows)
    print("-" * 66)
    print(f"{ok}/{len(rows)} backbones bind into the scaffold")

    outp = ROOT / a.json
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(dict(scene=str(d), views=len(paths),
                                    results=rows), indent=2))
    print(f"report -> {outp}")


if __name__ == "__main__":
    main()
