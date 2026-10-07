#!/usr/bin/env python3
"""Train / evaluate the sequence NVS decoder on cached maps (v202).

    python experiments/nvs_seq_decoder.py train --caches 'cache/nvsseq/vggt_omega/train/*' --method raw --out ckpt/nvsseq/vggt_omega_raw.pt
    python experiments/nvs_seq_decoder.py train --caches 'cache/nvsseq/vggt_omega/train/*' --method smr --out ckpt/nvsseq/vggt_omega_smr.pt
    python experiments/nvs_seq_decoder.py eval  --caches 'cache/nvsseq/vggt_omega/test/*' --ckpt-raw ckpt/nvsseq/vggt_omega_raw.pt \
        --ckpt-smr ckpt/nvsseq/vggt_omega_smr.pt --out outputs/nvs_seq_v2/vggt_omega

Training: a target is an input frame of a training sequence; its sources are the --sources nearest other input frames by
GT camera (distance + angle), excluding the target itself and the --exclude keyframes on each side of it (the held-out test targets have
their nearest inputs at distance 1, so training keeps the same geometry regime), so the target's own pixels are
never in the sources.  Geometry: the method's placed points (raw chain or +SMR), normalised per sample (centre of the source
cameras, median source depth = 1).  Loss: L1 + 0.5 LPIPS + 0.2 (1 - SSIM) at the native aspect.  One decoder per
geometry; the architecture, data, schedule and seed are identical, so a difference between the two systems is a
difference in the geometry they were given.
Evaluation: held-out targets of the test caches, sources = nearest inputs by GT camera (identical for every method);
raw is rendered with the raw decoder, smr / smr_pgo / gt with the SMR decoder (or --ckpt for all); a target is
"revisited" when its sources come from windows at least 3 apart (two visits).
"""
from __future__ import annotations

import argparse, glob, json, math, pathlib, random, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ------------------------------------------------------------------ data
def from7(v):
    from scipy.spatial.transform import Rotation
    v = np.asarray(v, float)
    return float(v[0]), Rotation.from_rotvec(v[1:4]).as_matrix(), v[4:7]


def apply_pose(S, T):
    s, R, t = S
    out = np.eye(4); out[:3, :3] = R @ T[:3, :3]; out[:3, 3] = s * (R @ T[:3, 3]) + t
    return out


def inv7(S):
    s, R, t = S
    return 1.0 / s, R.T, -(R.T @ t) / s


class Cache:
    def __init__(self, d):
        d = pathlib.Path(d); self.d = d
        self.meta = json.load(open(d / "meta.json"))
        L = lambda k: np.load(d / f"{k}.npy", mmap_mode="r")
        for k in ("pts_raw", "pts_smr", "rgb_src", "conf", "K_src", "owner", "in_gt", "tgt_gt", "K_gt", "tgt_kf", "in_kf"):
            setattr(self, k, L(k))
        self.S = {m: np.load(d / f"S_{m}.npy") for m in ("raw", "smr", "smr_pgo", "gt")}
        self.est = {m: np.load(d / f"est_{m}.npy") for m in ("raw", "smr", "smr_pgo", "gt")}
        self.E2G = {m: from7(np.load(d / f"E2G_{m}.npy")) for m in ("raw", "smr", "smr_pgo", "gt")}
        self.in_paths = json.load(open(d / "in_paths.json")); self.tgt_paths = json.load(open(d / "tgt_paths.json"))
        self.n = len(self.in_paths); self.H, self.W = int(self.meta["H"]), int(self.meta["W"])

    def nearest(self, T, k, exclude_center=None, exclude=0):
        c = self.in_gt[:, :3, 3]; step = float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1))) + 1e-9
        dist = np.linalg.norm(c - T[:3, 3], axis=1) / step
        ang = np.degrees(np.arccos(np.clip(self.in_gt[:, :3, 2] @ T[:3, 2], -1, 1)))
        score = dist + ang / 15.0
        if exclude_center is not None:
            score[max(0, exclude_center - exclude): exclude_center + exclude + 1] = np.inf
        return [int(j) for j in np.argsort(score)[:k] if np.isfinite(score[j])]


def load_image(path, H, W):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS), np.float32) / 255.0


def build_sample(C, method, srcs, T_gt_target, conf_keep=0.7, edge_thresh=0.05):
    """Sources of one target -> network inputs (numpy) and the target camera, all in the sample-normalised frame."""
    pts = C.pts_raw if method == "raw" else C.pts_smr
    P_all, cams, rgbs, cfs, Ks = [], [], [], [], []
    for j in srcs:
        s, R, t = from7(C.S[method][j])
        X = np.asarray(pts[j], np.float32).reshape(-1, 3)
        P_all.append((s * (R @ X.T).T + t).reshape(C.H, C.W, 3))
        cams.append(C.est[method][j][:3, 3]); rgbs.append(np.asarray(C.rgb_src[j], np.float32) / 255.0)
        cfs.append(np.asarray(C.conf[j], np.float32)); Ks.append(np.asarray(C.K_src[j], float))
    P = np.stack(P_all); cams = np.stack(cams)
    centre = cams.mean(0)
    dep = np.linalg.norm(P - cams[:, None, None, :], axis=-1)
    scale = float(np.nanmedian(dep)) if np.isfinite(dep).any() else 1.0
    Pn = (P - centre) / scale; depn = dep / scale; camn = (cams - centre) / scale
    ray = (Pn - camn[:, None, None, :]) / np.maximum(depn[..., None], 1e-6)
    valid = np.isfinite(Pn).all(-1) & np.isfinite(depn)
    cf = np.stack(cfs)
    for v in range(len(srcs)):
        m = valid[v]
        if conf_keep < 1.0 and m.any():
            m &= cf[v] >= np.quantile(cf[v][m], 1.0 - conf_keep)
        if edge_thresh > 0:
            dz = np.nan_to_num(depn[v], nan=0.0); dd = np.zeros_like(dz)
            for ax in (0, 1):
                df = np.abs(np.diff(dz, axis=ax))
                for padw in ((0, 1), (1, 0)):
                    pad = [(0, 0), (0, 0)]; pad[ax] = padw; dd = np.maximum(dd, np.pad(df, pad))
            m &= dd <= edge_thresh * np.maximum(dz, 1e-6)
        valid[v] = m
    foot = depn / np.array([K[0, 0] for K in Ks])[:, None, None]
    x = np.concatenate([np.stack(rgbs), Pn, depn[..., None], np.log1p(np.maximum(cf, 0))[..., None], ray], -1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # target camera: GT pose -> method map (inverse of the map->GT fit) -> sample frame
    Tm = apply_pose(inv7(C.E2G[method]), np.asarray(T_gt_target, float))
    Tm[:3, 3] = (Tm[:3, 3] - centre) / scale
    return dict(x=x, xyz=np.nan_to_num(Pn, nan=0.0).astype(np.float32), rgb=np.stack(rgbs).astype(np.float32),
                foot=np.nan_to_num(foot, nan=1e-3).astype(np.float32), valid=valid, c2w=Tm.astype(np.float32))


def forward(dec, ref, smp, K, H, W, device):
    import torch
    from smr.nvs.seq_decoder import render_hw
    t = lambda a: (a if torch.is_tensor(a) else torch.from_numpy(np.asarray(a))).to(device)
    x = t(smp["x"]).permute(0, 3, 1, 2); xyz = t(smp["xyz"]).permute(0, 3, 1, 2); rgb = t(smp["rgb"]).permute(0, 3, 1, 2)
    foot = t(smp["foot"])[:, None]; valid = t(smp["valid"])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        g = dec(x, xyz, rgb, foot, valid)
    g = {k: v.float() for k, v in g.items()}
    img, alpha, depth = render_hw(g, t(smp["c2w"])[None], t(np.asarray(K, np.float32)), H, W, bg=0.0)
    if ref is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            img = ref(img, alpha, (depth / 5.0).clamp(0, 1)).float()
    return img


# ------------------------------------------------------------------ train
class SampleStream:
    """Infinite stream of training samples; instantiated inside each DataLoader worker (own caches, own RNG)."""

    def __init__(self, dirs, method, sources, exclude, conf_keep, edge_thresh, seed):
        self.dirs, self.method, self.sources, self.exclude, self.conf_keep, self.edge_thresh, self.seed = dirs, method, sources, exclude, conf_keep, edge_thresh, seed

    def __iter__(self):
        import torch
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + 1000 * (info.id if info else 0))
        caches = [Cache(d) for d in self.dirs]
        while True:
            C = rng.choice(caches); ti = rng.randrange(C.n)
            srcs = C.nearest(C.in_gt[ti], self.sources, exclude_center=ti, exclude=self.exclude)
            if len(srcs) < 2:
                continue
            smp = build_sample(C, self.method, srcs, C.in_gt[ti], self.conf_keep, self.edge_thresh)
            if smp["valid"].sum() < 1000:
                continue
            smp["gt"] = load_image(C.in_paths[ti], C.H, C.W); smp["K"] = np.asarray(C.K_gt, np.float32); smp["H"] = C.H; smp["W"] = C.W
            yield smp


def train(a):
    import torch
    from smr.nvs.seq_decoder import SeqGaussianDecoder, Refiner
    from smr.nvs.gaussian_head import Perceptual, ssim
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    caches = [Cache(d) for d in sorted(glob.glob(a.caches)) if (pathlib.Path(d) / "meta.json").exists()]
    if not caches:
        sys.exit(f"no caches match {a.caches}")
    print(f"{len(caches)} training sequences, {sum(c.n for c in caches)} frames; geometry = {a.method}", flush=True)
    dec = SeqGaussianDecoder(base=a.base).to(device); ref = Refiner(base=a.ref_base).to(device) if a.refine else None
    params = list(dec.parameters()) + (list(ref.parameters()) if ref else [])
    print(f"parameters: decoder {sum(p.numel() for p in dec.parameters()) / 1e6:.1f}M" + (f", refiner {sum(p.numel() for p in ref.parameters()) / 1e6:.1f}M" if ref else ""), flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    perc = Perceptual().to(device)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    class _DS(torch.utils.data.IterableDataset):
        def __init__(self, stream): super().__init__(); self.stream = stream
        def __iter__(self): return iter(self.stream)
    stream = SampleStream([str(c.d) for c in caches], a.method, a.sources, a.exclude, a.conf_keep, a.edge_thresh, a.seed)
    loader = torch.utils.data.DataLoader(_DS(stream), batch_size=None, num_workers=a.workers, prefetch_factor=4 if a.workers else None,
                                         persistent_workers=bool(a.workers))
    it = iter(loader)
    t0 = time.time(); run = []
    for step in range(a.steps):
        loss_acc = 0.0
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            smp = next(it)
            gt = (smp["gt"] if torch.is_tensor(smp["gt"]) else torch.from_numpy(smp["gt"])).permute(2, 0, 1)[None].to(device).float()
            img = forward(dec, ref, smp, smp["K"], int(smp["H"]), int(smp["W"]), device)
            loss = (img - gt).abs().mean() + 0.5 * perc(img, gt) + 0.2 * (1 - ssim(img, gt))
            (loss / a.accum).backward(); loss_acc += float(loss) / a.accum
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        run.append(loss_acc)
        if (step + 1) % a.log_every == 0:
            print(f"step {step + 1}/{a.steps}  loss {np.mean(run[-a.log_every:]):.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f} s", flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            torch.save(dict(dec=dec.state_dict(), ref=ref.state_dict() if ref else None, args=vars(a), step=step + 1), out)
    print("saved", out)


# ------------------------------------------------------------------ eval
def evaluate(a):
    import torch
    from smr.nvs.seq_decoder import SeqGaussianDecoder, Refiner
    from smr.nvs.gaussian_head import Perceptual, ssim
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perc = Perceptual().to(device)

    def load(path):
        ck = torch.load(path, map_location=device, weights_only=False); ar = ck["args"]
        d = SeqGaussianDecoder(base=ar.get("base", 48)).to(device).eval(); d.load_state_dict(ck["dec"])
        r = None
        if ck.get("ref"):
            r = Refiner(base=ar.get("ref_base", 32)).to(device).eval(); r.load_state_dict(ck["ref"])
        return d, r, ar
    models = {}
    for m in a.methods:
        path = a.ckpt or (a.ckpt_raw if m == "raw" else a.ckpt_smr)
        models[m] = load(path)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allrec = {m: [] for m in a.methods}
    for d in sorted(glob.glob(a.caches)):
        if not (pathlib.Path(d) / "meta.json").exists():
            continue
        C = Cache(d); name = C.d.name
        if len(C.tgt_gt) == 0:
            continue
        srcs_of = [C.nearest(np.asarray(T, float), a.sources) for T in C.tgt_gt]
        revisit = [(max(int(C.owner[j]) for j in s) - min(int(C.owner[j]) for j in s)) >= 2 for s in srcs_of]
        line = f"{name:<28} ({len(C.tgt_gt)} targets, {sum(revisit)} revisited, {C.meta['closures']} closures)"
        for m in a.methods:
            dec, ref, ar = models[m]
            for ti, T in enumerate(C.tgt_gt):
                smp = build_sample(C, m, srcs_of[ti], np.asarray(T, float), ar.get("conf_keep", 0.7), ar.get("edge_thresh", 0.05))
                gt = load_image(C.tgt_paths[ti], C.H, C.W)
                with torch.no_grad():
                    img = forward(dec, ref, smp, C.K_gt, C.H, C.W, device)
                gtt = torch.from_numpy(gt).permute(2, 0, 1)[None].to(device)
                mse = float(((img - gtt) ** 2).mean())
                rec = dict(seq=name, target_kf=int(C.tgt_kf[ti]), revisit=bool(revisit[ti]), psnr=10 * math.log10(1 / max(mse, 1e-10)),
                           ssim=float(ssim(img, gtt)), lpips=float(perc(img, gtt)))
                allrec[m].append(rec)
                if a.save_images and ti in set(np.linspace(0, len(C.tgt_gt) - 1, a.save_images).astype(int)):
                    (out / "images" / name).mkdir(parents=True, exist_ok=True)
                    Image.fromarray((img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")).save(out / "images" / name / f"{m}_t{int(C.tgt_kf[ti]):03d}.png")
                    Image.fromarray((gt * 255).astype("uint8")).save(out / "images" / name / f"gt_t{int(C.tgt_kf[ti]):03d}.png")
            v = [r for r in allrec[m] if r["seq"] == name]
            line += f" | {m} {np.mean([r['psnr'] for r in v]):.2f}"
        print(line, flush=True)
    summ = {}
    print(f"\n{'method':<8} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} | revisited {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6}  (n)")
    for m, R in allrec.items():
        A = np.array([[r["psnr"], r["ssim"], r["lpips"]] for r in R]); V = np.array([[r["psnr"], r["ssim"], r["lpips"]] for r in R if r["revisit"]])
        mA = A.mean(0) if len(A) else np.full(3, np.nan); mV = V.mean(0) if len(V) else np.full(3, np.nan)
        summ[m] = dict(all=mA.tolist(), revisited=mV.tolist(), n=len(A), n_revisited=len(V))
        print(f"{m:<8} {mA[0]:>6.2f} {mA[1]:>6.3f} {mA[2]:>6.3f} | revisited {mV[0]:>6.2f} {mV[1]:>6.3f} {mV[2]:>6.3f}  ({len(V)})")
        with open(out / f"{m}.jsonl", "w") as f:
            for r in R:
                f.write(json.dumps(r) + "\n")
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print("->", out / "summary.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--caches", required=True); t.add_argument("--method", default="raw", choices=["raw", "smr", "smr_pgo"]); t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=12000); t.add_argument("--accum", type=int, default=2); t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--warmup", type=int, default=500); t.add_argument("--sources", type=int, default=8); t.add_argument("--exclude", type=int, default=1, help="keyframes on each side of the target removed from its sources (test sources are at distance >= 1)")
    t.add_argument("--base", type=int, default=48); t.add_argument("--ref-base", type=int, default=32); t.add_argument("--refine", type=int, default=1)
    t.add_argument("--conf-keep", type=float, default=0.7); t.add_argument("--edge-thresh", type=float, default=0.05)
    t.add_argument("--workers", type=int, default=6, help="DataLoader workers preparing samples (0 = in the main process)")
    t.add_argument("--log-every", type=int, default=100); t.add_argument("--save-every", type=int, default=2000); t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--caches", required=True); e.add_argument("--out", required=True)
    e.add_argument("--ckpt", default=None); e.add_argument("--ckpt-raw", default=None); e.add_argument("--ckpt-smr", default=None)
    e.add_argument("--methods", nargs="+", default=["raw", "smr", "smr_pgo", "gt"]); e.add_argument("--sources", type=int, default=8)
    e.add_argument("--save-images", type=int, default=4)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
