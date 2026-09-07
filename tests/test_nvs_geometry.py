"""Geometry-module test with a stub backbone (no torch, no Blender).

Synthetic object: a sphere of radius 0.4 at the origin, "rendered" analytically from the
protocol cameras into RGBA PNGs (alpha = hit mask) with a cams.json.  The stub backbone
returns GT geometry in its own frame (a random Sim(3) of the GT frame, scale 3) with
ordering-dependent depth noise and a 518^2 grid.  Checks: (1) placement through the GT
cameras recovers the sphere surface (scale recovered from the cameras); (2) the read
(4 orderings) reduces the geometric error of the raw pass; (3) head_input shapes/NaN handling.
"""
import json, pathlib, sys, tempfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smr.nvs import cameras as C  # noqa: E402
from smr.nvs import geometry as G  # noqa: E402

R_SPHERE = 0.4


def sphere_depth(K, c2w, res):
    ys, xs = np.mgrid[0:res, 0:res]
    pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs)], -1).reshape(-1, 3).astype(float)
    d = (c2w[:3, :3] @ (np.linalg.inv(K) @ pix.T)).T                 # world direction per unit camera-z
    c = c2w[:3, 3]
    a = (d * d).sum(1); b = 2 * (d @ c); cc = c @ c - R_SPHERE ** 2
    disc = b * b - 4 * a * cc
    z = np.full(len(pix), np.nan)
    hit = disc > 0
    z[hit] = (-b[hit] - np.sqrt(disc[hit])) / (2 * a[hit])
    return z.reshape(res, res)


def make_object(tmp, n_views=6, res=64):
    from PIL import Image
    views = C.gso_views(seed=0)[:n_views]; K = C.intrinsics(res)
    for i, v in enumerate(views):
        z = sphere_depth(K, v["c2w"], res)
        rgba = np.zeros((res, res, 4), np.uint8); hit = np.isfinite(z)
        rgba[..., 0] = 200; rgba[..., 1] = 80; rgba[..., 2] = 40; rgba[..., 3] = hit * 255
        Image.fromarray(rgba, "RGBA").save(tmp / f"{i:03d}.png")
    json.dump(C.cams_to_json(views, K, res), open(tmp / "cams.json", "w"))
    return views, K


class StubBackbone:
    """GT geometry expressed in a random Sim(3) frame (scale 3), on a 518^2 grid, with
    ordering-dependent depth noise (so orderings disagree and the read has work to do)."""

    def __init__(self, tmp, views, K_render, res_render, noise=0.02):
        self.tmp, self.views, self.noise = tmp, views, noise
        self.res_bb = 518; self.K_bb = C.intrinsics(self.res_bb)
        rng = np.random.default_rng(7)
        q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
        Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        self.S = np.eye(4); self.S[:3, :3] = 3.0 * Rm; self.S[:3, 3] = rng.normal(size=3)   # frame: X_bb = S X_gt
        self.calls = 0

    def infer(self, paths):
        self.calls += 1
        ids = [int(pathlib.Path(p).stem) for p in paths]
        rng = np.random.default_rng(1000 + sum(i * 10 ** k for k, i in enumerate(ids)))   # ordering-dependent
        poses, depth = [], []
        for i in ids:
            c2w = np.asarray(self.views[i]["c2w"]); poses.append(self.S @ c2w @ np.diag([1, 1, 1, 1.0]))
            z = sphere_depth(self.K_bb, c2w, self.res_bb) * 3.0                            # depth scales with the frame
            z = z * (1 + self.noise * rng.normal(size=z.shape))
            depth.append(z)
        poses = np.stack(poses)
        poses[:, :3, :3] /= 3.0                                                            # rotation part must stay orthonormal
        conf = np.stack([np.where(np.isfinite(d), 5.0, 0.0) for d in depth])

        class Out:  # minimal BackboneOutput look-alike
            pass
        o = Out(); o.poses = poses; o.depth = np.stack(depth); o.extras = dict(conf=conf)
        return o


def surface_error(pts):
    r = np.linalg.norm(pts.reshape(-1, 3), axis=1)
    r = r[np.isfinite(r)]
    return np.abs(r - R_SPHERE).mean(), len(r)


def test_placement_and_read():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp); res = 64
        views, K = make_object(tmp, n_views=4, res=res)
        d = G.load_views(tmp)
        assert d["rgb"].shape == (4, res, res, 3) and d["alpha"].max() == 1.0 and d["rgb"][0][d["alpha"][0] == 0].min() == 1.0
        bb = StubBackbone(tmp, views, K, res, noise=0.03)
        raw = G.predict_geometry(bb, d["paths"], d["c2w"], d["K"], res, reads=1)
        assert abs(raw["scales"][0] - 1 / 3.0) < 1e-6, raw["scales"]                      # metric scale from the cameras
        e_raw, n_raw = surface_error(raw["pts"])
        read = G.predict_geometry(bb, d["paths"], d["c2w"], d["K"], res, reads=4, tau_abs=0.05)
        e_read, n_read = surface_error(read["pts"])
        print(f"surface error raw {e_raw*1000:.1f} mm ({n_raw} pts) -> read x4 {e_read*1000:.1f} mm ({n_read} pts); backbone calls {bb.calls}")
        assert e_raw < 0.03 and e_read < 0.75 * e_raw and n_read >= 0.95 * n_raw and bb.calls == 5
        x, valid = G.head_input(d["rgb"], read["pts"], read["conf"], d["alpha"])
        assert x.shape == (4, res, res, 8) and np.isfinite(x).all() and valid.sum() > 0
        assert (x[..., 7][valid] > 0.5).all() and (x[..., 3:6][~valid] == 0).all()


if __name__ == "__main__":
    test_placement_and_read(); print("nvs geometry tests passed")
