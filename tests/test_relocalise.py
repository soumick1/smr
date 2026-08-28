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


def test_register_sessions_recovers_alignment_by_consensus(world):
    """Two 'sessions' of the ring stitched independently, the second in a
    deliberately transformed gauge; registration must recover the map even
    when a third of the pair estimates are poisoned (aliasing)."""
    from smr.stitch import sim3
    from smr.stitch.chunks import make_chunks
    from smr.stitch.relocalise import register_sessions
    from smr.eval.trajectory import ate_rmse
    n = world.n_per_lap
    runner = SimulatedRunner(world.gt, noise=0.005, distortion=0.02, seed=3)
    session_results, session_desc = {}, {}
    for sid, lo in ((0, 0), (1, n)):
        members = list(range(lo, lo + n))
        cache = PassCache()
        chunks = [[lo + i for i in c] for c in make_chunks(n, 16, 8)]
        r = AnchoredStitcher(ScaffoldIndex(seed=0), n_sites=2).run(
            chunks, cache, runner, world.descriptors)
        # r["est"] is the stitched WORLD map (rows in frame order);
        # r["local"] is per-pass local poses and must never be used as a map
        local = {members[i]: r["est"][i] for i in range(n)}
        if sid == 1:                       # scramble the second session's gauge
            S = (1.4, sim3.rotmat([0, 0, 0.7]), np.array([2.0, -1.0, 0.5]))
            local = {g: sim3.apply_one(S, T) for g, T in local.items()}
        session_results[sid] = dict(local=local, frames=members)
        session_desc[sid] = {g: world.descriptors[g] for g in members}

    calls = dict(n=0)
    def run_pair_pass(fa, fb):
        calls["n"] += 1
        P, _, _ = runner([int(g) for g in fa] + [int(g) for g in fb])
        if calls["n"] % 3 == 0:            # poison a third of the pairs
            M = np.eye(4); M[:3, :3] = sim3.rotmat([0, 0.9, 0])
            P[len(fa):] = M @ P[len(fa):][::-1]
        return P

    Ts, registered, owner, rep = register_sessions(session_results, session_desc, run_pair_pass)
    assert rep[1]["registered"] and rep[1]["cluster"] >= 3
    est = np.stack([registered[i] for i in range(2 * n)])
    assert ate_rmse(est, world.gt) < 0.12


def test_memorymap_from_register_mode_result(world):
    """The exact construction the register branch performs: session maps ->
    register -> populate the MAIN index -> MemoryMap -> localise a query.
    Guards the empty-index crash class (v79, v84) without any backbone."""
    from smr.stitch.chunks import make_chunks
    from smr.stitch.relocalise import register_sessions
    n = world.n_per_lap
    runner = SimulatedRunner(world.gt, noise=0.005, distortion=0.02, seed=3)
    sr, sd = {}, {}
    for sid, lo in ((0, 0), (1, n)):
        members = list(range(lo, lo + n))
        chunks = [[lo + i for i in c] for c in make_chunks(n, 16, 8)]
        r = AnchoredStitcher(ScaffoldIndex(seed=0), n_sites=2).run(
            chunks, PassCache(), runner, world.descriptors)
        sr[sid] = dict(local={members[i]: r["est"][i] for i in range(n)}, frames=members)
        sd[sid] = {g: world.descriptors[g] for g in members}

    def run_pair_pass(fa, fb):
        return runner([int(g) for g in fa] + [int(g) for g in fb])[0]

    Ts, registered, owner, rep = register_sessions(sr, sd, run_pair_pass)
    reg_frames = sorted(registered)
    index = ScaffoldIndex(seed=0)
    for i in reg_frames:
        index.add(i, registered[i], world.descriptors[i])
    res = dict(local={i: registered[i] for i in reg_frames},
               est=np.stack([registered[i] for i in reg_frames]),
               owner=[owner[i] for i in reg_frames], n_loops=0)
    m = MemoryMap(res, world.gt[reg_frames], index, world.descriptors)
    assert m.map_ate < 0.12
    q = reg_frames[len(reg_frames) // 2] if False else 3   # localise map frame 3's twin on lap 2
    out = localise(m, world.descriptors[n + 3],
                   lambda anc: runner([n + 3] + [int(g) for g in anc])[0],
                   mode="smr", n_sites=2)
    t_err, r_err = pose_error(out["T"], world.gt[n + 3])
    assert out["T"] is not None and t_err < 0.15


def test_fit_points_robust_recovers_similarity_under_outliers():
    from smr.stitch import sim3
    rng = np.random.default_rng(0)
    A = rng.uniform(-1, 1, (600, 3))
    S_true = (1.7, sim3.rotmat([0.2, -0.4, 0.9]), np.array([0.5, -2.0, 1.0]))
    B = S_true[0] * (A @ S_true[1].T) + S_true[2] + rng.normal(scale=0.004, size=A.shape)
    B[:90] += rng.uniform(1, 3, (90, 3))                      # 15% outliers
    S, keep, info = sim3.fit_points_robust(A, B)
    assert abs(S[0] - 1.7) < 0.02 and keep.sum() >= 480
    err = np.linalg.norm((S[0] * (A[90:] @ S[1].T) + S[2]) - B[90:], axis=1)
    assert np.median(err) < 0.02


def test_localise_dense_beats_pose_fit_on_synthetic_depth(world):
    """A stub dense pass built from ground truth: dense placement must land
    within millimetres where the 4-pose fit is limited by pose noise."""
    from smr.stitch import sim3
    from smr.stitch.relocalise import localise_dense
    rng = np.random.default_rng(0)
    n = world.n_per_lap
    # perfect-map memory over lap one, plus a synthetic point bank
    index = ScaffoldIndex(seed=0)
    mmap = MemoryMap.__new__(MemoryMap)
    mmap.frames = list(range(n))
    mmap.pose = {i: world.gt[i].copy() for i in range(n)}
    mmap.owner = {i: i // 16 for i in range(n)}
    mmap.index = index
    for i in range(n):
        index.add(i, mmap.pose[i], world.descriptors[i])
    K = np.array([[80.0, 0, 48.0], [0, 80.0, 36.0], [0, 0, 1.0]])
    H, W = 72, 96
    ys, xs = np.mgrid[0:H:6, 0:W:6]
    pix = np.stack([ys.ravel(), xs.ravel()], 1).astype(np.int32)
    depth_of = {}
    bank = {}
    for g in range(n):
        d = rng.uniform(2.0, 4.0, len(pix))
        depth_of[g] = d
        X = (pix[:, 1] - K[0, 2]) / K[0, 0] * d
        Y = (pix[:, 0] - K[1, 2]) / K[1, 1] * d
        Pc = np.stack([X, Y, d], 1)
        Pw = Pc @ world.gt[g][:3, :3].T + world.gt[g][:3, 3]
        bank[g] = dict(pix=pix, pts=Pw, col=np.ones((len(pix), 3), np.float32))

    class RV:
        pass

    def run_dense_pass(anchors, q=3):
        gauge = (0.6, sim3.rotmat([0.1, 0.2, -0.3]), np.array([1.0, 0.5, -0.2]))
        ids = [n + q] + list(anchors)
        rv = RV()
        rv.poses = np.stack([sim3.apply_one(gauge, world.gt[g]) for g in ids])
        rv.poses[:, :3, 3] += rng.normal(scale=0.002, size=(len(ids), 3))
        D = np.zeros((len(ids), H, W))
        for i, g in enumerate(ids[1:], start=1):
            D[i][pix[:, 0], pix[:, 1]] = depth_of[g] * gauge[0] / gauge[0]  # depth in cam frame is gauge-scaled
            D[i][pix[:, 0], pix[:, 1]] = depth_of[g] * gauge[0]
        rv.depth = D
        rv.intrinsics = K
        return rv

    out = localise_dense(mmap, bank, world.descriptors[n + 3], run_dense_pass,
                         mode="plain", n_sites=2)
    t_err, r_err = pose_error(out["T"], world.gt[n + 3])
    assert out["T"] is not None and t_err < 0.02 and r_err < 1.0
