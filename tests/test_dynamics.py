import numpy as np
from smr.dynamics import Ring, Torus2D

def test_ring_width_and_place():
    r = Ring(seed=1); r.settle(700)
    assert abs(r.width() - r.bump_width) / r.bump_width < 0.10
    for tgt in (0.3, 2.0, 5.1):
        err = abs(((r.place(tgt) - tgt + np.pi) % (2 * np.pi)) - np.pi)
        assert err < 0.01

def test_calibration_sign_and_tracking():
    r = Ring(seed=2)
    slope = r.calibrate()
    assert slope > 0
    r.settle(200); th0 = r.decode()
    for _ in range(400):
        r.step(0.10)
    moved = ((r.decode() - th0) % (2 * np.pi))
    assert abs(moved - 0.10 * 400 * r.dt) < 0.15

def test_torus_place():
    t = Torus2D(seed=3)
    px, py = t.place((2.0, 4.5))
    assert abs(((px - 2.0 + np.pi) % (2 * np.pi)) - np.pi) < 0.02
    assert abs(((py - 4.5 + np.pi) % (2 * np.pi)) - np.pi) < 0.02
