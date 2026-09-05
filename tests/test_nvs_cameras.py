"""Camera protocol sanity: look-at geometry, projection of the origin, Pluecker rays, GSO layout."""
import sys, pathlib, json, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from smr.nvs import cameras as C


def test_lookat_and_projection():
    K = C.intrinsics()
    for v in C.train_views(50, seed=3) + C.gso_views(seed=1):
        c2w = v["c2w"]
        assert np.allclose(c2w[:3, :3] @ c2w[:3, :3].T, np.eye(3), atol=1e-9) and np.linalg.det(c2w[:3, :3]) > 0
        assert abs(np.linalg.norm(c2w[:3, 3]) - C.RADIUS) < 1e-9
        uv, z = C.project(K, c2w, np.zeros((1, 3)))
        assert np.allclose(uv[0], [C.RES / 2, C.RES / 2]) and abs(z[0] - C.RADIUS) < 1e-9   # origin at the image centre, at distance R
        # the +y world axis (up) projects ABOVE the centre (smaller v) when the camera is level or above
        uv_up, _ = C.project(K, c2w, np.array([[0, 0.2, 0]]))
        if v["el"] > -10:
            assert uv_up[0, 1] < C.RES / 2
    # the bounding sphere (radius 0.5) fits with margin: 200 surface points inside the image from every view
    rng = np.random.default_rng(0); S = rng.normal(size=(200, 3)); S = 0.5 * S / np.linalg.norm(S, axis=1, keepdims=True)
    for v in C.gso_views(seed=0) + C.train_views(20, seed=5):
        uv, z = C.project(K, v["c2w"], S)
        assert (uv > 20).all() and (uv < C.RES - 20).all() and (z > 1.0).all(), (v["az"], v["el"], uv.min(), uv.max())


def test_pluecker_and_json():
    K = C.intrinsics(); v = C.gso_views(seed=0)[0]
    P = C.pluecker_rays(K, v["c2w"])
    d, m = P[..., :3], P[..., 3:]
    assert np.allclose(np.linalg.norm(d, axis=-1), 1) and np.allclose((d * m).sum(-1), 0, atol=1e-9)    # Pluecker constraint d.m = 0
    centre_dir = d[C.RES // 2, C.RES // 2]
    assert np.dot(centre_dir, -v["c2w"][:3, 3] / C.RADIUS) > 0.999                                       # centre ray points at the origin
    js = json.dumps(C.cams_to_json(C.gso_views(seed=0), K))
    d2 = json.loads(js); assert len(d2["views"]) == 14 and sum(x["role"] == "input" for x in d2["views"]) == 4


if __name__ == "__main__":
    test_lookat_and_projection(); test_pluecker_and_json(); print("nvs camera tests passed")
