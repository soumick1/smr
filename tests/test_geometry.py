import numpy as np
from smr.utils.geometry import (euler_zyx_to_R, R_to_euler_zyx, wrap_pi,
                                make_T, inv_T, pose_errors)

def test_euler_roundtrip():
    g = np.random.default_rng(0)
    for _ in range(50):
        y, p, r = g.uniform(-np.pi, np.pi), g.uniform(-1.3, 1.3), \
                  g.uniform(-np.pi, np.pi)
        y2, p2, r2 = R_to_euler_zyx(euler_zyx_to_R(y, p, r))
        assert abs(wrap_pi(y - y2)) < 1e-9
        assert abs(p - p2) < 1e-9 and abs(wrap_pi(r - r2)) < 1e-9

def test_se3_inverse():
    g = np.random.default_rng(1)
    T = make_T(euler_zyx_to_R(*g.uniform(-1, 1, 3)), g.uniform(-2, 2, 3))
    assert np.allclose(inv_T(T) @ T, np.eye(4), atol=1e-10)
    re, pe = pose_errors(T, T)
    assert re < 1e-9 and pe < 1e-12
