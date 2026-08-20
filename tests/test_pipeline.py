import numpy as np
from smr.backbones import get_backbone
from smr.dynamics import ScaffoldState
from smr.pipeline import bind, rotate, coherent_reanchor

def test_bind_rotate_smoke():
    bb = get_backbone("synthetic", seed=0)
    out = bb.infer(n_views=4)
    ss = ScaffoldState(periods=[2.4, 3.2], ring_N=64, torus_N=24, seed=0,
                       omega_max=0.16)          # skip vmax sweep for speed
    ss.calibrate()
    bound = bind(out, ss, bb.cam, formation_extent=1.2,
                 formation_spacing=0.6, torus_N=24, N_h=512, k=48)
    ss.place_pose(out.poses[0])
    res = rotate(bound, out.poses[1], bb.cam, k_recall=2, settle_fiber=False)
    assert res["mask"].mean() > 0.10
    assert np.isfinite(res["xi"]).all()
    assert res["n_steps"] < 6000


def test_reanchor_noop_when_clean():
    bb = get_backbone("synthetic", seed=0)
    out = bb.infer(n_views=3)
    ss = ScaffoldState(periods=[2.4, 3.2], ring_N=64, torus_N=24, seed=0,
                       omega_max=0.16)
    ss.calibrate()
    bound = bind(out, ss, bb.cam, formation_extent=1.2,
                 formation_spacing=0.6, torus_N=24, N_h=512, k=48)
    ss.place_pose(out.poses[0])
    d = coherent_reanchor(bound)
    assert d["corrected"] is False
    assert d["score_before"] <= bound.novelty_tau
