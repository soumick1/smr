"""CPU dry run of the v122 pipeline: points_suite (geo filter + maps) -> ETH3D
point-map evaluator, on a synthetic tilted plane with known ground truth.

Checks: (1) the geometric-consistency filter removes injected outliers so the
fused cloud is closer to the plane than the unfiltered single cloud;
(2) maps.npz has the documented contents; (3) the evaluator's Umeyama step
brings the deliberately mis-scaled backbone frame back to the GT plane.
"""
import json, pathlib, subprocess, sys, tempfile, types
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

import points_suite as PS  # noqa: E402
from smr.backbones.base import BackboneOutput  # noqa: E402
from smr.eval.eth3d_gt import fisheye_project  # noqa: E402  (unused, import check)


def look_at(cam, target, up=(0, 1, 0.1)):
    z = target - cam; z = z / np.linalg.norm(z)
    up = np.asarray(up, float); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, cam
    return M


def plane_depth(K, c2w, H, W):
    ys, xs = np.mgrid[0:H, 0:W]
    rays = np.linalg.inv(K) @ np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5, np.ones(H * W)])
    d_w = c2w[:3, :3] @ rays
    return ((0.0 - c2w[2, 3]) / d_w[2]).reshape(H, W)


class FakeBackbone:
    """Exact plane depth at a 1/8 grid of the 'image', in a frame scaled by 1/0.8
    with 4 % of pixels turned into gross outliers in every view."""
    def __init__(self, gt, Hd=48, Wd=64, bias=False, gross=False):
        self.gt = gt; self.Hd, self.Wd = Hd, Wd; self.bias = bias; self.gross = gross

    def infer(self, image_paths):
        ids = [self.gt["path_to_id"][p] for p in image_paths]
        # ordering-dependent noise (like a real backbone: the first frame fixes
        # the frame) and, if enabled, a per-pass point-vs-camera scale bias
        # (0 %, +2 %, +4 % by which frame comes first) -- the two in-window
        # errors v124's --reads / --content-align are meant to correct
        rng = np.random.default_rng(ids[0])
        b = 0.0 if not self.bias else (0.0 if ids[0] == 0 else (0.02 if ids[0] < 4 else 0.04))
        Wu, Hu = self.gt["wh"]
        depths, Ks, poses, confs, wps, wpc = [], [], [], [], [], []
        s = 1.25                                        # backbone gauge: metric x 1.25
        for v in ids:
            K = self.gt["K"][v].copy(); K[0] *= self.Wd / Wu; K[1] *= self.Hd / Hu
            c2w = self.gt["poses"][v]
            d_clean = plane_depth(K, c2w, self.Hd, self.Wd) * (1.0 + b)     # biased depth, exact cameras
            if self.gross and ids[0] != 0:                                  # a read that fails grossly on the left half
                d_clean[:, : self.Wd // 2] *= 1.30
            out = rng.random(d_clean.shape) < 0.04
            d = d_clean * s; d[out] *= rng.uniform(1.1, 1.4, out.sum())
            depths.append(d); Ks.append(K)
            p = c2w.copy(); p[:3, 3] *= s; poses.append(p)
            confs.append(np.where(out, 0.5, 1.0))
            # point-map head: exact plane points in the (scaled) world frame, with its own
            # outliers and expp1-style confidences (3.0 good, 1.5 bad) for the --conf-abs test
            ys, xs = np.mgrid[0:self.Hd, 0:self.Wd]
            rays = np.linalg.inv(K) @ np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5, np.ones(xs.size)])  # same centres as plane_depth
            dd = d_clean                                        # point head from the clean geometry
            Xc = rays * dd.ravel(); Xw = (c2w[:3, :3] @ Xc).T + c2w[:3, 3]
            Xw = Xw * s
            out2 = rng.random(dd.shape) < 0.04
            Xw = Xw.reshape(self.Hd, self.Wd, 3); Xw[out2] += rng.normal(0, 0.3, (out2.sum(), 3))
            wps.append(Xw); wpc.append(np.where(out2, 1.5, 3.0))
        return BackboneOutput(poses=np.stack(poses), intrinsics=Ks[0], depth=np.stack(depths),
                              rgb=np.zeros((len(ids), self.Hd, self.Wd, 3)), mask=np.ones((len(ids), self.Hd, self.Wd), bool),
                              extras=dict(conf=np.stack(confs), scene_scale=1.0, intrinsics_all=np.stack(Ks),
                                          world_points=np.stack(wps), world_points_conf=np.stack(wpc)))


def make_gt(tmp, V=8, Wu=512, Hu=384):
    from PIL import Image
    tmp = pathlib.Path(tmp)
    (tmp / "img").mkdir(); (tmp / "scene" / "dslr_scan_eval").mkdir(parents=True)
    f = 480.0
    K = np.array([[f, 0, Wu / 2], [0, f, Hu / 2], [0, 0, 1.0]])
    paths, poses, Ks = [], [], []
    for v in range(V):
        ang = 2 * np.pi * v / V
        cam = np.array([1.2 * np.cos(ang), 1.2 * np.sin(ang), 3.0])
        c2w = look_at(cam, np.array([0.1 * np.cos(ang), 0.1 * np.sin(ang), 0.0]))
        p = tmp / "img" / f"v{v:02d}.jpg"; Image.new("RGB", (Wu, Hu)).save(p)
        paths.append(str(p)); poses.append(c2w); Ks.append(K)
    npz = tmp / "plane.npz"
    np.savez(npz, image_paths=np.array(paths), poses=np.stack(poses), K=np.stack(Ks), scene="plane")
    # synthetic 'laser scan' of the plane + mlp
    rng = np.random.default_rng(1)
    P = np.stack([rng.uniform(-3, 3, 300000), rng.uniform(-3, 3, 300000), np.zeros(300000)], -1)
    ply = tmp / "scene" / "dslr_scan_eval" / "scan1.ply"
    with open(ply, "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n" % len(P))
        np.savetxt(fh, P, fmt="%.4f")
    (tmp / "scene" / "dslr_scan_eval" / "scan_alignment.mlp").write_text(
        '<!DOCTYPE MeshLabDocument>\n<MeshLabProject>\n<MeshGroup>\n<MLMesh label="scan1.ply" filename="scan1.ply">\n'
        '<MLMatrix44>\n1 0 0 0 \n0 1 0 0 \n0 0 1 0 \n0 0 0 1 \n</MLMatrix44>\n</MLMesh>\n</MeshGroup>\n</MeshLabProject>\n')
    return dict(npz=npz, paths=paths, poses=np.stack(poses), K=np.stack(Ks), wh=(Wu, Hu),
                path_to_id={p: i for i, p in enumerate(paths)}, scene=tmp / "scene")


def make_pilot(tmp, gt):
    """Synthetic pilot_a --save-est npz: 3 chunks of 4 over 8 keyframes; 'smr' row
    exact, 'chained' row with a drift of 0.1 m per two frames along world z."""
    key = np.arange(8); chunks = [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]]
    est_smr = gt["poses"].copy()
    est_ch = gt["poses"].copy()
    for i in range(8):
        est_ch[i, 2, 3] += 0.1 * (i // 2)
    out = pathlib.Path(tmp) / "pilot.npz"
    np.savez_compressed(out, gt=str(gt["npz"]), backbone="fake", key=key,
                        chunks=np.array([np.asarray(c) for c in chunks], dtype=object),
                        kpaths=np.array(gt["paths"], dtype=object),
                        rows=np.array(["chained", "smr"], dtype=object), est_chained=est_ch, est_smr=est_smr)
    return out


def run_suite(gt, out_dir, extra, bias=False, gross=False):
    args = ["--gt", str(gt["npz"]), "--backbone", "fake", "--w", "8", "--overlap", "0", "--k-ctx", "1",
            "--stride", "1", "--out-dir", str(out_dir), "--save-maps"] + extra
    a = PS.build_parser().parse_args(args)
    import smr.backbones as B
    B.get_backbone = lambda name, **kw: FakeBackbone(gt, bias=bias, gross=gross)   # monkeypatch registry lookup
    PS.real(a)
    return a


def plane_err(ply):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dtu_eval_mod", ROOT / "scripts" / "dtu_eval.py")
    de = importlib.util.module_from_spec(spec); spec.loader.exec_module(de)
    P = de.read_ply(ply)
    return (np.abs(P[:, 2]).mean() if len(P) else 0.0), len(P)


def test_dryrun():
    with tempfile.TemporaryDirectory() as tmp:
        gt = make_gt(tmp)
        raw = pathlib.Path(tmp) / "raw"; geo = pathlib.Path(tmp) / "geo"
        run_suite(gt, raw, [])
        run_suite(gt, geo, ["--geo-views", "3", "--geo-src", "5"])
        e_raw, n_raw = plane_err(raw / "single.ply"); e_geo, n_geo = plane_err(geo / "single.ply")
        print(f"mean |z| raw {e_raw:.4f} m ({n_raw:,} pts)  geo {e_geo:.4f} m ({n_geo:,} pts)")
        assert e_geo < 0.25 * e_raw, "geometric consistency did not remove the injected outliers"
        assert n_geo > 0.6 * n_raw, "filter too aggressive on clean pixels"
        # point-map head route with the absolute confidence cut (VGGT-p, C > 2.0)
        ph = pathlib.Path(tmp) / "ph"; run_suite(gt, ph, ["--points-from", "pointhead", "--conf-abs", "2.0"])
        e_ph, n_ph = plane_err(ph / "single.ply")
        ph_raw = pathlib.Path(tmp) / "ph_raw"; run_suite(gt, ph_raw, ["--points-from", "pointhead"])
        e_phr, n_phr = plane_err(ph_raw / "single.ply")
        print(f"pointhead: raw |z| {e_phr:.4f} m ({n_phr:,})  conf>2 |z| {e_ph:.4f} m ({n_ph:,})")
        assert e_ph < 1e-6 and 0.94 < n_ph / n_phr < 0.98, (e_ph, n_ph, n_phr)
        # downstream matrix path: same passes placed under two pose rows
        pilot = make_pilot(tmp, gt)
        errs = {}
        for row in ("chained", "smr"):
            o = pathlib.Path(tmp) / f"pilot_{row}"
            run_suite(gt, o, ["--pilot", str(pilot), "--row", row, "--points-from", "pointhead", "--conf-abs", "2.0"])
            errs[row] = plane_err(o / "fused.ply")[0]
        print(f"pilot rows: chained |z| {errs['chained']:.4f} m   smr |z| {errs['smr']:.4f} m")
        assert errs["smr"] < 1e-6 and errs["chained"] > 0.02, errs
        # v124: in-window memory operations on a biased backbone (0/+2/+4 % per pass)
        r1 = pathlib.Path(tmp) / "reads";  run_suite(gt, r1, ["--reads", "3", "--points-from", "pointhead", "--conf-abs", "2.0"], bias=True)
        # --content-ref first keeps read 0's scale (exact in this fake) -> exactness checks;
        # the default symmetric reference must land at the mean bias instead (checked below)
        r2 = pathlib.Path(tmp) / "reads_ca"; run_suite(gt, r2, ["--reads", "3", "--points-from", "pointhead", "--conf-abs", "2.0", "--content-align", "--content-ref", "first"], bias=True)
        r4 = pathlib.Path(tmp) / "reads_ca_sim3"; run_suite(gt, r4, ["--reads", "3", "--points-from", "pointhead", "--conf-abs", "2.0", "--content-align", "--content-model", "sim3"], bias=True)
        r3 = pathlib.Path(tmp) / "reads_ca_gt"; run_suite(gt, r3, ["--reads", "3", "--points-from", "pointhead", "--conf-abs", "2.0", "--content-align", "--gt-cams", "--content-ref", "first"], bias=True)
        r6 = pathlib.Path(tmp) / "reads_ca_mean"; run_suite(gt, r6, ["--reads", "3", "--points-from", "pointhead", "--conf-abs", "2.0", "--content-align"], bias=True)
        e6 = plane_err(r6 / "fused.ply")[0]
        print(f"symmetric reference: fused |z| {e6:.4f} m (expected ~2% of ~3 m depth = mean of the 0/2/4% biases)")
        assert 0.035 < e6 < 0.09, e6
        e1, e2, e3 = plane_err(r1 / "fused.ply")[0], plane_err(r2 / "fused.ply")[0], plane_err(r3 / "fused.ply")[0]
        e_single = plane_err(r1 / "single.ply")[0]
        e4 = plane_err(r4 / "fused.ply")[0]
        # v125: (i) with --content-ref first the reference read's bias survives; with mean it does not.
        # Read 0 of the biased fake is exact, so both are ~0 here -- instead check the symmetric
        # renormalisation is applied and that the fused cloud with fallback keeps at least as many
        # points as the reference read (a read is never emptier than a single pass).
        M2 = np.load(r2 / "maps.npz", allow_pickle=True)
        n_single = np.isfinite(M2["single_maps"]).all(-1).sum(); n_fused = np.isfinite(M2["fused_maps"]).all(-1).sum()
        print(f"fallback: single read {n_single:,} finite pixels, fused {n_fused:,}")
        assert n_fused >= n_single, (n_single, n_fused)
        r5 = pathlib.Path(tmp) / "reads_nofb"
        run_suite(gt, r5, ["--reads", "3", "--points-from", "pointhead", "--conf-abs", "2.0", "--content-align", "--content-ref", "first", "--fallback", "none"], bias=True)
        M5 = np.load(r5 / "maps.npz", allow_pickle=True)
        assert np.isfinite(M5["fused_maps"]).all(-1).sum() <= n_fused
        print(f"reads=3 fused |z|: camera placement {e1:.4f}  +content-align(scale) {e2:.4f}  +content-align(gt-cams) {e3:.4f}  "
              f"+content-align(sim3) {e4:.4f}   (single read {e_single:.4f})")
        assert e2 < 0.25 * e1 and e2 < 0.005 and e3 < 0.005 and e4 < 0.25 * e1, (e1, e2, e3, e4)
        # v127: two reads, the second failing grossly (x1.3 depth) on the left half of every image.
        # With 2 witnesses nothing reaches m=2 where they differ: modest disagreement (the 2 % bias
        # right half) -> median fallback keeps the pixel; gross disagreement (left half) -> abstained.
        g1 = pathlib.Path(tmp) / "gross_abstain"
        run_suite(gt, g1, ["--reads", "2", "--points-from", "pointhead", "--conf-abs", "2.0", "--fallback", "median"], bias=True, gross=True)
        Mg = np.load(g1 / "maps.npz", allow_pickle=True); fm = Mg["fused_maps"]
        left, right = np.isfinite(fm[:, :, : fm.shape[2] // 2]).all(-1).mean(), np.isfinite(fm[:, :, fm.shape[2] // 2:]).all(-1).mean()
        print(f"abstention: finite fraction left (gross) {left:.2f}  right (modest) {right:.2f}")
        assert left < 0.15 and right > 0.9, (left, right)
        g2 = pathlib.Path(tmp) / "gross_noabstain"
        run_suite(gt, g2, ["--reads", "2", "--points-from", "pointhead", "--conf-abs", "2.0", "--fallback", "median", "--abstain-rel", "0"], bias=True, gross=True)
        fm2 = np.load(g2 / "maps.npz", allow_pickle=True)["fused_maps"]
        assert np.isfinite(fm2).all(-1).mean() > 0.9, "abstain-rel 0 must never abstain"
        M = np.load(geo / "maps.npz", allow_pickle=True)
        assert M["single_maps"].shape == (8, 48, 64, 3) and M["fused_maps"].shape[0] == 8
        assert list(M["view_ids_in_gt"]) == list(range(8))
        # evaluator on the scan fallback; the backbone frame was metric so the
        # pre-align numbers are already small and Umeyama must not break them
        cmd = [sys.executable, str(ROOT / "scripts" / "eth3d_pointmap_eval.py"), "--maps", str(geo / "maps.npz"),
               "--scene-dir", str(gt["scene"]), "--gt", "scan", "--json", str(pathlib.Path(tmp) / "r.jsonl")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        print(r.stdout)
        res = json.loads((pathlib.Path(tmp) / "r.jsonl").read_text().splitlines()[-1])
        assert res["umeyama"]["overall"] < 0.02, res["umeyama"]
        assert abs(res["umeyama_scale"] - 1.0) < 0.02


if __name__ == "__main__":
    test_dryrun()
    print("points_suite dry run passed")
