import numpy as np
from smr.memory import ToyVectorHash, RLSMemory, BlockScaffold, Novelty

def test_toy_all_fixed():
    tv = ToyVectorHash([3, 4, 5], N_h=160, seed=0).form()
    assert tv.frac_fixed() == 1.0

def test_rls_exact():
    m = RLSMemory(64, 16)
    g = np.random.default_rng(2)
    H, S = g.standard_normal((64, 10)), g.standard_normal((16, 10))
    for i in range(10):
        m.write(H[:, i], S[:, i])
    assert np.linalg.norm(m.read_s(H[:, 3]) - S[:, 3]) < 1e-2

def test_novelty_separation():
    bs = BlockScaffold([2.4, 3.2], torus_N=16, N_h=256, k=24, seed=0)
    bs.form_grid(extent=0.5, spacing=0.5)
    nov = Novelty().fit(bs)
    g = np.random.default_rng(3)
    fam = [nov.score(bs._formed_g[:, i]) for i in range(5)]
    randoms = [nov.score(g.standard_normal(bs.N_g)) for _ in range(5)]
    assert max(fam) < 0.05 and min(randoms) > 0.5
