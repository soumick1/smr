#!/usr/bin/env python3
"""Train the completion head on harvested (splat -> frame) pairs.

    python experiments/train_completion.py --data data/completion \
        --name shakedown --val-scenes room,orbit_frames

Screen-safe by design: everything is logged to
outputs/logs/completion/<name>/  --  train.log (timestamped, mirrors the
console), metrics.csv / eval.csv (machine-readable), config.json (args +
git commit + dataset statistics), samples/ (qualitative grids at every
eval), ckpt_last.pt / ckpt_best.pt.  Re-running with --resume continues
from ckpt_last after a disconnect.

Splits are SCENE-level (never pair-level: neighbouring views of one
scene would leak).  Torch is imported lazily so --dry-run exercises the
full data/split/logging pipeline on any machine."""
import argparse, csv, hashlib, json, logging, math, pathlib, random, re
import subprocess, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PAT = re.compile(r"_h\d{3}(_c(?:all|\d{2}))?\.npz$")


def scene_of(fn):
    stem = PAT.sub("", fn)               # {scene}_{backbone}
    return stem.rsplit("_", 1)[0]


class PairDataset:
    """Pure-numpy dataset over harvested npz pairs (torch-free)."""

    def __init__(self, files, crop=None, hole_bias=2, seed=0):
        self.files = list(files)
        self.crop, self.hole_bias = crop, hole_bias
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _load(fp):
        d = np.load(fp)
        sr = d["splat_rgb"].astype(np.float32)
        sd = np.clip(d["splat_depth"].astype(np.float32), 0, 8)
        sm = d["splat_mask"].astype(np.float32)
        tr = d["tgt_rgb"].astype(np.float32)
        td = d["tgt_depth"].astype(np.float32)
        tv = (d["tgt_mask"] & (td > 0)).astype(np.float32)
        x = np.concatenate([sr, sd[..., None] / 4.0, sm[..., None]], -1)
        return x, sr, sd, sm, tr, td, tv

    def _crop(self, arrs, H, W):
        c = self.crop
        pads = [(max(0, c - H), max(0, c - W))]
        if pads[0][0] or pads[0][1]:
            ph, pw = pads[0]
            arrs = [np.pad(a, ((0, ph), (0, pw)) + ((0, 0),) * (a.ndim - 2),
                           mode="reflect") for a in arrs]
            H, W = arrs[0].shape[:2]
        best = None
        for _ in range(max(1, self.hole_bias)):
            y = self.rng.randint(0, H - c); xx = self.rng.randint(0, W - c)
            cov = arrs[3][y:y + c, xx:xx + c].mean()   # splat mask coverage
            if best is None or cov < best[0]:
                best = (cov, y, xx)
        _, y, xx = best
        return [a[y:y + c, xx:xx + c] for a in arrs]

    def get(self, i):
        arrs = list(self._load(self.files[i]))
        H, W = arrs[0].shape[:2]
        if self.crop:
            arrs = self._crop(arrs, H, W)
        x, sr, sd, sm, tr, td, tv = arrs
        return dict(x=np.ascontiguousarray(x.transpose(2, 0, 1)),
                    splat_rgb=sr.transpose(2, 0, 1), splat_depth=sd[None],
                    splat_mask=sm[None], tgt_rgb=tr.transpose(2, 0, 1),
                    tgt_depth=td[None], tgt_valid=tv[None])

    def batches(self, bs, shuffle=True, epoch=0):
        idx = list(range(len(self)))
        if shuffle:
            random.Random(1000 + epoch).shuffle(idx)
        for k in range(0, len(idx) - bs + 1 if shuffle else len(idx), bs):
            items = [self.get(j) for j in idx[k:k + bs]]
            yield {key: np.stack([it[key] for it in items])
                   for key in items[0]}


def split_scenes(files, val_scenes, val_frac):
    scenes = sorted({scene_of(f.name) for f in files})
    if val_scenes:
        vs = set(val_scenes.split(","))
        missing = vs - set(scenes)
        assert not missing, f"--val-scenes not in data: {missing}"
    else:
        vs = {s for s in scenes
              if int(hashlib.md5(s.encode()).hexdigest(), 16) % 1000
              < val_frac * 1000}
    tr = [f for f in files if scene_of(f.name) not in vs]
    va = [f for f in files if scene_of(f.name) in vs]
    return tr, va, sorted(vs), scenes


def dataset_stats(files):
    cov, ctx = [], {}
    for f in files[: min(len(files), 400)]:
        d = np.load(f)
        cov.append(float(d["splat_mask"].mean()))
        c = int(d["context"]) if "context" in d else -1
        ctx[c] = ctx.get(c, 0) + 1
    return dict(n=len(files),
                coverage_mean=round(float(np.mean(cov)), 3),
                coverage_min=round(float(np.min(cov)), 3),
                coverage_max=round(float(np.max(cov)), 3),
                context_hist={str(k): v for k, v in sorted(ctx.items())})


def make_logger(run):
    lg = logging.getLogger("completion"); lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(run / "train.log"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh)
    return lg


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True,
                              text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def save_samples(run, step, rows, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for j, r in enumerate(rows):
        fig, axs = plt.subplots(1, 6, figsize=(16.5, 3.0))
        panels = [("splat", r["splat"]), ("prediction", r["pred"]),
                  ("composite", r["comp"]), ("target", r["tgt"]),
                  ("|err| composite", r["err"]), ("depth comp.", r["dcomp"])]
        if "name" in r:
            fig.suptitle(r["name"], fontsize=8, y=1.02)
        for ax, (t, im) in zip(axs, panels):
            ax.imshow(im if im.ndim == 3 else im, cmap=None if im.ndim == 3
                      else ("magma" if "err" in t else "viridis"))
            ax.set_title(t, fontsize=9); ax.axis("off")
        fig.tight_layout()
        fig.savefig(run / "samples" / f"step{step:06d}_{tag}{j}.png",
                    dpi=110, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out-root", default="outputs/logs/completion")
    ap.add_argument("--name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--val-scenes", default="")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--hole-weight", type=float, default=2.0)
    ap.add_argument("--perc-weight", type=float, default=0.0,
                    help="VGG16 perceptual loss on the composite (0=off). "
                         "Recommended 0.05 for corpus-scale training; the "
                         "pass-through rule focuses its gradients on holes.")
    ap.add_argument("--depth-weight", type=float, default=0.5)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all-val", action="store_true",
                    help="treat every pair as validation (no training "
                         "split); for zero-shot --eval-only on a corpus "
                         "the checkpoint never trained on")
    a = ap.parse_args()

    run = ROOT / a.out_root / a.name
    (run / "samples").mkdir(parents=True, exist_ok=True)
    lg = make_logger(run)

    files = sorted(sum([list(pathlib.Path(d).glob("*.npz"))
                        for d in a.data], []))
    assert files, f"no npz pairs under {a.data}"
    tr_f, va_f, val_scenes, scenes = split_scenes(files, a.val_scenes,
                                                  a.val_frac)
    if a.all_val:
        tr_f, va_f, val_scenes = [], list(files), scenes
    assert va_f and (tr_f or a.eval_only or a.dry_run),         "empty split -- adjust --val-scenes/--val-frac (--all-val needs "         "--eval-only or --dry-run)"
    st_tr = dataset_stats(tr_f) if tr_f else {"n": 0}
    st_va = dataset_stats(va_f)
    cfg = dict(args=vars(a), git=git_commit(),
               scenes=scenes, val_scenes=val_scenes,
               train_stats=st_tr, val_stats=st_va)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    lg.info(f"run dir: {run}")
    lg.info(f"pairs: {len(files)} | train {len(tr_f)} / val {len(va_f)} | "
            f"scenes {len(scenes)} (val: {','.join(val_scenes)})")
    if tr_f:
        lg.info(f"train coverage {st_tr['coverage_mean']} "
                f"[{st_tr['coverage_min']}, {st_tr['coverage_max']}] | "
                f"context hist {st_tr['context_hist']}")
    else:
        lg.info(f"all-val mode: {len(va_f)} evaluation pairs, "
                f"coverage {st_va['coverage_mean']}")

    if a.dry_run:
        pool = tr_f or va_f
        ds = PairDataset(pool, crop=a.crop, seed=a.seed)
        b = next(ds.batches(min(a.batch, len(pool))))
        lg.info("DRY RUN: sample batch " +
                ", ".join(f"{k}{tuple(v.shape)}" for k, v in b.items()))
        lg.info("DRY RUN OK -- data, split, logging verified"); return

    import torch                                   # lazy: GPU box only
    from smr.completion import CompletionUNet
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) \
        else "cpu"
    model = CompletionUNet(base=a.base).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    lg.info(f"torch {torch.__version__} | device {dev} "
            f"({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})"
            f" | params {n_par/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    perc = None
    if a.perc_weight > 0:
        try:
            import torchvision
            try:        # torchvision >= 0.13
                _w = torchvision.models.VGG16_Weights.IMAGENET1K_V1
                _vgg_full = torchvision.models.vgg16(weights=_w)
            except AttributeError:   # older torchvision API
                _vgg_full = torchvision.models.vgg16(pretrained=True)
            vgg = _vgg_full.features[:16].to(dev).eval()
            for p in vgg.parameters():
                p.requires_grad_(False)
            _m = torch.tensor([0.485, 0.456, 0.406],
                              device=dev).view(1, 3, 1, 1)
            _s = torch.tensor([0.229, 0.224, 0.225],
                              device=dev).view(1, 3, 1, 1)
            perc = lambda x: vgg((x - _m) / _s)      # noqa: E731
            lg.info(f"perceptual loss on (VGG16 relu3_3, w={a.perc_weight})")
        except Exception as e:
            lg.warning(f"perceptual loss unavailable ({e}); continuing "
                       f"with L1 only")
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / max(1, a.warmup)
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    step, best = 0, -1e9
    if a.resume and (run / "ckpt_last.pt").exists():
        ck = torch.load(run / "ckpt_last.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        lg.info(f"resumed from step {step} (best {best:.3f})")
    if a.eval_only:
        ck = torch.load(a.eval_only, map_location=dev)
        model.load_state_dict(ck["model"])
        lg.info(f"eval-only from {a.eval_only} (step {ck['step']})")

    def to_t(b):
        return {k: torch.from_numpy(v).to(dev, non_blocking=True)
                for k, v in b.items()}

    def losses(b):
        rgb, dep = model(b["x"])
        hole = 1.0 - b["splat_mask"]
        w = 1.0 + (a.hole_weight - 1.0) * hole
        l_rgb = (w * (rgb - b["tgt_rgb"]).abs()).mean()
        l_rgb_hole = ((hole * (rgb - b["tgt_rgb"]).abs()).sum()
                      / (3 * hole.sum() + 1e-6))
        vd = b["tgt_valid"]
        l_dep = ((w * vd * (dep - b["tgt_depth"]).abs()).sum()
                 / (vd.sum() + 1e-6)) * a.depth_weight
        l_perc = torch.zeros((), device=l_rgb.device)
        if perc is not None:
            comp = b["splat_mask"] * b["splat_rgb"] + hole * rgb
            l_perc = (perc(comp) - perc(b["tgt_rgb"])).abs().mean()                 * a.perc_weight
        return l_rgb + l_dep + l_perc, dict(
            loss_rgb=l_rgb, loss_rgb_hole=l_rgb_hole,
            loss_depth=l_dep, loss_perc=l_perc), (rgb, dep)

    @torch.no_grad()
    def evaluate(save_step=None):
        model.eval()
        ds = PairDataset(va_f, crop=None)
        agg = dict(psnr_all=[], psnr_hole=[], l1_hole=[],
                   absrel_hole=[], absrel_known=[])
        rows = []
        for i in range(len(ds)):
            it = ds.get(i)
            x = torch.from_numpy(it["x"][None]).to(dev)
            H, W = x.shape[-2:]
            ph, pw = (-H) % 16, (-W) % 16
            xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
            with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                rgb, dep = model(xp)
            rgb = rgb[..., :H, :W].float().cpu().numpy()[0]
            dep = dep[..., :H, :W].float().cpu().numpy()[0, 0]
            sm = it["splat_mask"][0]; tv = it["tgt_valid"][0] > 0
            comp = np.where(sm[None] > 0, it["splat_rgb"], rgb)
            dcomp = np.where(sm > 0, it["splat_depth"][0], dep)
            hole = (sm == 0)
            e2 = ((comp - it["tgt_rgb"]) ** 2)
            agg["psnr_all"].append(-10 * np.log10(e2.mean() + 1e-9))
            if hole.any():
                agg["psnr_hole"].append(
                    -10 * np.log10(e2[:, hole].mean() + 1e-9))
                agg["l1_hole"].append(
                    np.abs(comp - it["tgt_rgb"])[:, hole].mean())
            td = it["tgt_depth"][0]
            hv, kv = hole & tv, (~hole) & tv
            if hv.any():
                agg["absrel_hole"].append(
                    (np.abs(dcomp - td)[hv] / td[hv]).mean())
            if kv.any():
                agg["absrel_known"].append(
                    (np.abs(dcomp - td)[kv] / td[kv]).mean())
            if save_step is not None and len(rows) < 4 and hole.mean() > .05:
                err = np.abs(comp - it["tgt_rgb"]).mean(0)
                rows.append(dict(
                    name=pathlib.Path(ds.files[i]).name,
                    splat=it["splat_rgb"].transpose(1, 2, 0),
                    pred=rgb.transpose(1, 2, 0),
                    comp=comp.transpose(1, 2, 0),
                    tgt=it["tgt_rgb"].transpose(1, 2, 0),
                    err=err, dcomp=dcomp))
        model.train()
        m = {k: float(np.mean(v)) if v else float("nan")
             for k, v in agg.items()}
        if save_step is not None and rows:
            save_samples(run, save_step, rows)
        return m

    with open(run / "metrics.csv", "a", newline="") as fm, \
         open(run / "eval.csv", "a", newline="") as fe:
        mw = csv.writer(fm); ew = csv.writer(fe)
        if fm.tell() == 0:
            mw.writerow(["step", "lr", "loss", "loss_rgb", "loss_rgb_hole",
                         "loss_depth", "loss_perc", "grad_norm", "img_s"])
        if fe.tell() == 0:
            ew.writerow(["step", "psnr_all", "psnr_hole", "l1_hole",
                         "absrel_hole", "absrel_known", "is_best"])

        if a.eval_only:
            m = evaluate(save_step=step)
            lg.info("EVAL " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
            return

        ds = PairDataset(tr_f, crop=a.crop, seed=a.seed)
        t0, seen, epoch = time.time(), 0, 0
        lg.info(f"training: {a.steps} steps, batch {a.batch}, "
                f"crop {a.crop}, lr {a.lr} (cosine, warmup {a.warmup})")
        while step < a.steps:
            epoch += 1
            for b in ds.batches(a.batch, epoch=epoch):
                if step >= a.steps:
                    break
                step += 1
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                bt = to_t(b)
                with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                    loss, parts, _ = losses(bt)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                seen += a.batch
                if step % a.log_every == 0:
                    ips = seen / (time.time() - t0)
                    eta = (a.steps - step) * a.batch / max(ips, 1e-9) / 60
                    lg.info(
                        f"step {step}/{a.steps} ep{epoch} "
                        f"lr {lr_at(step):.2e} loss {loss.item():.4f} "
                        f"(rgb {parts['loss_rgb'].item():.4f} "
                        f"hole {parts['loss_rgb_hole'].item():.4f} "
                        f"dep {parts['loss_depth'].item():.4f} "
                        f"perc {parts['loss_perc'].item():.4f}) "
                        f"gn {float(gn):.2f} | {ips:.1f} img/s eta {eta:.0f}m")
                    mw.writerow([step, f"{lr_at(step):.3e}",
                                 f"{loss.item():.5f}",
                                 f"{parts['loss_rgb'].item():.5f}",
                                 f"{parts['loss_rgb_hole'].item():.5f}",
                                 f"{parts['loss_depth'].item():.5f}",
                                 f"{parts['loss_perc'].item():.5f}",
                                 f"{float(gn):.3f}", f"{ips:.2f}"])
                    fm.flush()
                if step % a.eval_every == 0 or step == a.steps:
                    m = evaluate(save_step=step)
                    is_best = m["psnr_hole"] > best
                    if is_best:
                        best = m["psnr_hole"]
                        torch.save(dict(model=model.state_dict(),
                                        opt=opt.state_dict(), step=step,
                                        best=best, args=vars(a)),
                                   run / "ckpt_best.pt")
                    torch.save(dict(model=model.state_dict(),
                                    opt=opt.state_dict(), step=step,
                                    best=best, args=vars(a)),
                               run / "ckpt_last.pt")
                    lg.info("EVAL step %d | %s%s" % (
                        step,
                        " ".join(f"{k}={v:.4f}" for k, v in m.items()),
                        "  ** best (ckpt_best.pt)" if is_best else ""))
                    ew.writerow([step] + [f"{m[k]:.5f}" for k in
                                          ("psnr_all", "psnr_hole", "l1_hole",
                                           "absrel_hole", "absrel_known")]
                                + [int(is_best)])
                    fe.flush()
        lg.info(f"done: best psnr_hole {best:.3f} | ckpts in {run}")


if __name__ == "__main__":
    main()
