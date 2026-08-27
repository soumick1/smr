"""Relocalisation on the rendered two-lap world: lap one is the map, lap two
the queries.  Every row places a query with the same pass machinery; the
rows differ only in where the anchors come from."""
import numpy as np
import pytest

from smr.stitch import AnchoredStitcher, DescriptorIndex, PassCache, ScaffoldIndex, SimulatedRunner
from smr.stitch.chunks import make_chunks
from smr.stitch.relocalise import MemoryMap, corrupt, localise, pose_error, summarise
from smr.stitch.simulate import LoopWorld


@pytest.fixture(scope="module")
def world():
    return LoopWorld(n_per_lap=40, laps=2, H=96, W=96, seed=0, jitter=0.02)


@pytest.fixture(scope="module")
def mapped(world):
    n = world.n_per_lap
    gt_map = world.gt[:n]
    chunks = make_chunks(n, 16, 8)
    cache = PassCache()
    runner = SimulatedRunner(world.gt, noise=0.005, distortion=0.02, seed=1)   # indices 0..n-1 are the map
    index = ScaffoldIndex(seed=0)
    res = AnchoredStitcher(index, n_sites=2).run(chunks, cache, runner, world.descriptors[:n])
    mmap = MemoryMap(res, gt_map, index, world.descriptors[:n])
    plain = DescriptorIndex()
    for g in mmap.frames:
        plain.add(g, mmap.pose[g], world.descriptors[g])
    mp = MemoryMap.__new__(MemoryMap); mp.__dict__.update(mmap.__dict__); mp.index = plain
    return mmap, mp, runner


def _run_rows(world, mapped, rows, desc=None):
    mmap, mp, runner = mapped
    n = world.n_per_lap
    out = {}
    for row in rows:
        errs = []
        for q in range(n, 2 * n):
            def run_pass(anchors, q=q):
                return runner([q] + list(anchors))[0]
            m = mp if row == "plain" else mmap
            d = (desc if desc is not None else world.descriptors)[q]
            r = localise(m, d, run_pass, mode=row, n_sites=2, gt_pose=world.gt[q], K=8)
            errs.append(pose_error(r["T"], world.gt[q]))
        out[row] = summarise(errs)
    return out


def test_memory_relocalises_where_recency_cannot(world, mapped):
    s = _run_rows(world, mapped, ["lastk", "plain", "smr", "oracle"])
    assert s["oracle"]["recall_10cm_10deg"] > 0.9           # placement works when retrieval is perfect
    assert s["smr"]["recall_10cm_10deg"] > 0.7
    assert s["plain"]["recall_10cm_10deg"] > 0.7             # predicted tie with the plain index
    assert s["lastk"]["recall_10cm_10deg"] < s["smr"]["recall_10cm_10deg"]   # recency is not a map


def test_map_is_metric_and_query_is_not_aligned(world, mapped):
    mmap, _, _ = mapped
    assert mmap.map_ate < 0.1
    # a query placed at a deliberately wrong pose must show its full error
    T = world.gt[world.n_per_lap + 3].copy(); T[0, 3] += 0.5
    t, r = pose_error(T, world.gt[world.n_per_lap + 3])
    assert abs(t - 0.5) < 1e-9 and r < 1e-6


def test_corruptions_are_well_formed():
    rng = np.random.default_rng(0)
    img = rng.random((48, 64, 3))
    for spec in ("gauss:0.1", "occlude:0.3", "blur:3", "dark:0.5", "none"):
        out = corrupt(img, spec, rng)
        assert out.shape == img.shape and out.min() >= 0 and out.max() <= 1
    assert corrupt(img, "blur:3", rng).std() < img.std()
    assert abs(corrupt(img, "occlude:0.25", rng).size - img.size) == 0
    assert np.allclose(corrupt(img, "dark:0.5", rng), img * 0.5)


def test_recall_degrades_gracefully_under_corruption(world, mapped):
    from smr.stitch.passes import array_descriptors
    rng = np.random.default_rng(0)
    n = world.n_per_lap
    clean = _run_rows(world, mapped, ["smr"])["smr"]["recall_10cm_10deg"]
    imgs = np.stack([corrupt(world.rgb[q], "gauss:0.15", rng) for q in range(2 * n)])
    desc = array_descriptors(imgs)
    noisy = _run_rows(world, mapped, ["smr"], desc=desc)["smr"]["recall_10cm_10deg"]
    assert noisy <= clean + 1e-9 and noisy > 0.3
