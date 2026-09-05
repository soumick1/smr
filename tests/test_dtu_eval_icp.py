"""dtu_eval.py ICP robustness (no data needed).

Builds a fake DTU scan: GT 'STL' = a bumpy sphere patch (mm), ObsMask/BB/Plane
.mat files in the official layout, and a prediction = GT + 0.3 mm noise
misaligned by scale 1.01 / 6 mm shift, plus 40 % 'table' points far below
the object (the situation that broke v125's runs).  The hardened ICP must
recover the alignment (Acc well below 1 mm) and must reject a runaway
refinement when the prediction is deliberately shifted by 80 mm.
"""
import pathlib, re, subprocess, sys, tempfile
import numpy as np
from scipy.io import savemat

ROOT = pathlib.Path(__file__).resolve().parents[1]


def write_ply(path, P):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n" % len(P))
        np.savetxt(f, P, fmt="%.4f")


def make_scan(tmp, shift_mm=6.0, scale=1.01, background=0.4, seed=0):
    rng = np.random.default_rng(seed)
    tmp = pathlib.Path(tmp)
    (tmp / "ss" / "ObsMask").mkdir(parents=True); (tmp / "stl").mkdir()
    # GT: upper hemisphere patch, radius 60 mm, centred at (0,0,650), dense (0.5 mm)
    n = 400000
    u = rng.uniform(-1, 1, n); th = rng.uniform(0, 2 * np.pi, n)
    r = 60.0 + 2.0 * np.sin(6 * th) * u
    z = np.abs(u) * r; rho = np.sqrt(np.maximum(r ** 2 - z ** 2, 0))
    gt = np.stack([rho * np.cos(th), rho * np.sin(th), 650.0 - z], -1)
    write_ply(tmp / "stl" / "stl001_total.ply", gt)
    # ObsMask: 10 mm grid over a BB with margin, everything inside the object's bbox observed
    Res = 10.0
    BB = np.array([[-100.0, -100.0, 560.0], [100.0, 100.0, 680.0]])
    shape = np.ceil((BB[1] - BB[0]) / Res).astype(int) + 1
    ObsMask = np.zeros(shape, dtype=np.uint8)
    lo = np.floor((gt.min(0) - 5 - BB[0]) / Res).astype(int); hi = np.ceil((gt.max(0) + 5 - BB[0]) / Res).astype(int)
    ObsMask[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = 1
    savemat(tmp / "ss" / "ObsMask" / "ObsMask1_10.mat", dict(ObsMask=ObsMask, BB=BB, Res=np.array([[Res]])))
    savemat(tmp / "ss" / "ObsMask" / "Plane1.mat", dict(P=np.array([0, 0, -1.0, 655.0])))   # keep z < 655
    # prediction: noisy GT, misaligned, plus background 'table' at z ~ 700 (below the object) and beyond the BB
    pred = gt[rng.choice(n, 150000, replace=False)] + rng.normal(0, 0.3, (150000, 3))
    pred = scale * pred + np.array([shift_mm, 0.0, 0.0])
    nb = int(background / (1 - background) * len(pred))
    table = np.stack([rng.uniform(-150, 150, nb), rng.uniform(-150, 150, nb), 700.0 + rng.normal(0, 0.5, nb)], -1)
    write_ply(tmp / "pred.ply", np.concatenate([pred, table]))
    return tmp


def run_eval(tmp, icp=50):
    cmd = [sys.executable, str(ROOT / "scripts" / "dtu_eval.py"), "--pred", str(tmp / "pred.ply"), "--scan", "1",
           "--sampleset", str(tmp / "ss"), "--points-dir", str(tmp / "stl"), "--icp", str(icp), "--down", "0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def parse(out):
    import re
    m = re.search(r"Acc ([\d.]+)  Comp ([\d.]+)  Overall ([\d.]+)", out)
    return [float(x) for x in m.groups()]


def test_background_does_not_capture_icp():
    with tempfile.TemporaryDirectory() as tmp:
        make_scan(tmp)
        out0 = run_eval(pathlib.Path(tmp), icp=0); out8 = run_eval(pathlib.Path(tmp), icp=50)
        print(out0.strip()); print(out8.strip())
        a0, a8 = parse(out0), parse(out8)
        assert "REJECTED" not in out8
        assert a8[0] < 0.45 and a8[0] < 0.2 * a0[0], (a0, a8)       # 6 mm / 1 % misalignment repaired to the 0.3 mm noise floor


def test_far_start_is_recovered_or_rejected():
    """An 80 mm shift: with the coarse centroid init (v128) the ICP recovers it to the
    noise floor; with --icp-init none the refinement is REJECTED, never a runaway."""
    with tempfile.TemporaryDirectory() as tmp:
        make_scan(tmp, shift_mm=80.0, scale=1.0)
        out = run_eval(pathlib.Path(tmp), icp=50)
        print(out.strip())
        assert "coarse" in out and parse(out)[0] < 0.45, parse(out)
        cmd = [sys.executable, str(ROOT / "scripts" / "dtu_eval.py"), "--pred", str(pathlib.Path(tmp) / "pred.ply"), "--scan", "1",
               "--sampleset", str(pathlib.Path(tmp) / "ss"), "--points-dir", str(pathlib.Path(tmp) / "stl"), "--icp", "50", "--down", "0", "--icp-init", "none"]
        r = subprocess.run(cmd, capture_output=True, text=True); print(r.stdout.strip())
        # without the coarse candidate an 80 mm start is never "fixed" by a runaway: either refused or left alone
        assert "REJECTED" in r.stdout or "keeping camera alignment" in r.stdout
        assert parse(r.stdout)[0] > 5.0


def test_large_scale_mismatch_is_recovered():
    """DUSt3R-like case: the cloud arrives 25 % too large and 40 mm off (point-vs-camera scale
    mismatch).  v134 refused this (5 % cap); the robust similarity initialisation + fit-quality
    selection must bring it to the noise floor."""
    with tempfile.TemporaryDirectory() as tmp:
        make_scan(tmp, shift_mm=40.0, scale=1.25)
        out = run_eval(pathlib.Path(tmp), icp=50); print(out.strip())
        assert "REJECTED" not in out.split("scan1: Acc")[0] or "coarse" in out
        assert parse(out)[0] < 0.45, parse(out)


def test_fit_quality_is_symmetric():
    """A prediction slid 30 mm along the surface must score worse than the true alignment."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("de", ROOT / "scripts" / "dtu_eval.py"); de = importlib.util.module_from_spec(spec); spec.loader.exec_module(de)
    with tempfile.TemporaryDirectory() as tmp:
        make_scan(tmp, shift_mm=0.0, scale=1.0); tmp = pathlib.Path(tmp)
        G = de.GT(1, tmp / "ss", tmp / "stl", "plane")
        P = de.read_ply(tmp / "pred.ply")
        good = G.fit_quality(P); slid = G.fit_quality(P + np.array([30.0, 0.0, 0.0]))
        # the scan13 failure mode: a cloud shrunk by 23 % about the object centre (and nudged 5 mm)
        c = np.median(P[G.in_region(P)], 0)
        shrunk = G.fit_quality(c + 0.77 * (P - c) + np.array([0.0, 0.0, 5.0]))
        print(f"fit true {good:.2f} mm, slid 30 mm {slid:.2f} mm, shrunk 23% {shrunk:.2f} mm")
        assert good < 0.5 * slid and good < 0.5 * shrunk, (good, slid, shrunk)


def write_ply_binary(path, P):
    P = np.asarray(P, np.float32)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n" % len(P)).encode())
        P.astype("<f4").tofile(f)


def test_multi_pred_tags_binary_and_equivalence():
    """Two clouds of one scan in one call (tags), one of them binary PLY; results match the
    single-call path, and the thinning/gauge match the v129 evaluator within noise."""
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        make_scan(tmp); tmp = pathlib.Path(tmp)
        P = np.loadtxt(tmp / "pred.ply", skiprows=7)
        write_ply_binary(tmp / "pred_bin.ply", P)
        cmd = [sys.executable, str(ROOT / "scripts" / "dtu_eval.py"), "--pred", str(tmp / "pred.ply"), str(tmp / "pred_bin.ply"),
               "--tags", "asc", "bin", "--scan", "1", "--sampleset", str(tmp / "ss"), "--points-dir", str(tmp / "stl"), "--icp", "50", "--down", "0.2"]
        r = subprocess.run(cmd, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-1500:]
        out = r.stdout; print(out.strip())
        assert out.count("== asc scan1 ==") == 1 and out.count("== bin scan1 ==") == 1
        res = re.findall(r"scan1: Acc ([\d.]+)  Comp ([\d.]+)  Overall ([\d.]+)", out)
        assert len(res) == 2
        a1, a2 = [float(x[2]) for x in res]
        assert abs(a1 - a2) < 0.005, (a1, a2)                    # float32 binary vs ascii: same cloud
        old = pathlib.Path("/tmp/dtu_eval_v129.py")
        if old.exists():                                          # equivalence with the previous evaluator
            r0 = subprocess.run([sys.executable, str(old), "--pred", str(tmp / "pred.ply"), "--scan", "1", "--sampleset", str(tmp / "ss"),
                                 "--points-dir", str(tmp / "stl"), "--icp", "50", "--down", "0.2"], capture_output=True, text=True)
            o0 = float(re.search(r"Overall ([\d.]+)", r0.stdout).group(1)); print("v129 evaluator:", r0.stdout.strip().splitlines()[-1])
            assert abs(o0 - a1) < 0.02, (o0, a1)


if __name__ == "__main__":
    test_background_does_not_capture_icp()
    test_far_start_is_recovered_or_rejected()
    test_large_scale_mismatch_is_recovered()
    test_fit_quality_is_symmetric()
    test_multi_pred_tags_binary_and_equivalence()
    print("dtu_eval icp tests passed")
