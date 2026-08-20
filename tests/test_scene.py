import numpy as np
from smr.scene import (Camera, build_room, camera_ring, render,
                       surfels_from_view, transform, splat)

def test_reprojection_consistency():
    cam = Camera(); pts, cols = build_room(seed=0)
    Ts = camera_ring(8)
    rgb, dep, msk = render(pts, cols, Ts[0], cam)
    assert msk.mean() > 0.4
    sp, sc = surfels_from_view(rgb, dep, msk, cam)
    Pw = transform(sp, Ts[0], Ts[1])
    rgb2, dep2, msk2 = splat(Pw, sc, Ts[1], cam)
    rgbG, depG, mskG = render(pts, cols, Ts[1], cam)
    both = msk2 & mskG
    assert both.mean() > 0.05
    assert np.median(np.abs(dep2[both] - depG[both])) < 0.06
