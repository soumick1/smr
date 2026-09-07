#!/usr/bin/env python3
"""Train the Gaussian head for one frozen backbone on our Objaverse renders.

    python experiments/train_nvs.py --backbone vggt --data ~/data/nvs/objaverse --out outputs/nvs/vggt --steps 30000
    python experiments/train_nvs.py --backbone vggt --data ~/data/nvs/objaverse --out outputs/nvs/smoke --smoke

Each step: one object, `--n-in` random input views + `--n-tgt` random targets; the frozen
backbone (raw pass, reads=1) gives the input geometry placed with the GT input cameras
(smr.nvs.geometry); the head turns valid pixels into Gaussians; gsplat renders the targets;
loss = MSE + lpips_w * LPIPS.  The head is the only trained module.  Screen-safe: train.log,
metrics.csv (step, loss, psnr, s/step), val.csv (held-out objects: PSNR/SSIM/LPIPS), config.json,
ckpt_last.pt / ckpt_best.pt; --resume continues from ckpt_last.  Held-out objects: a fixed 1 %
by id hash (never trained on).  --smoke: 20 steps, 2 val objects, prints device/timings.
"""
import argparse, csv, json, pathlib, random, subprocess, sys, time, zlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def list_objects(data):
    root = pathlib.Path(data).expanduser()
    objs = sorted(p.parent for p in root.glob("*/cams.json"))
    excl = set((root / "exclude.txt").read_text().split()) if (root / "exclude.txt").exists() else set()
    return [o for o in objs if not (o / "FAILED").exists() and o.name not in excl]


def is_val(obj_dir, frac=0.01):
    return (zlib.crc32(obj_dir.name.encode()) % 10000) < frac * 10000


def to_torch(x, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


def step_batch(bb, head, G, d, in_ids, tgt_ids, res, device, reads, source):
    """Geometry (numpy) -> head -> rendered targets.  Returns pred (T,3,H,W), gt (T,3,H,W), n_gaussians."""
    import torch
    from smr.nvs.gaussian_head import render
    geo = G.predict_geometry(bb, [d["paths"][i] for i in in_ids], d["c2w"][in_ids], d["K"], res, reads=reads, source=source)
    x, valid = G.head_input(d["rgb"][in_ids], geo["pts"], geo["conf"], d["alpha"][in_ids])
    xt = to_torch(x, device).permute(0, 3, 1, 2)
    g = head(xt, xt[:, 3:6], xt[:, 0:3], to_torch(valid, device))
    if g["means"].shape[0] == 0:
        return None, None, 0
    pred, _ = render(g, to_torch(d["c2w"][tgt_ids], device), to_torch(d["K"], device), res)
    gt = to_torch(d["rgb"][tgt_ids], device).permute(0, 3, 1, 2)
    return pred, gt, int(g["means"].shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--data", default="~/data/nvs/objaverse")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-in", type=int, default=4); ap.add_argument("--n-tgt", type=int, default=4)
    ap.add_argument("--lpips", type=float, default=0.1, help="LPIPS weight (0 disables; saves the VGG load)")
    ap.add_argument("--reads", type=int, default=1, help="training geometry: 1 = raw pass (default)")
    ap.add_argument("--source", default="auto", choices=["auto", "depth", "pointhead"])
    ap.add_argument("--val-every", type=int, default=2000); ap.add_argument("--val-objects", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone-kw", action="append", default=[])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.val_every, a.val_objects, a.ckpt_every = 20, 10, 2, 10

    import torch
    from smr.backbones import get_backbone
    from smr.nvs import geometry as G
    from smr.nvs.gaussian_head import GaussianHead, Perceptual, psnr, ssim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    objs = list_objects(a.data)
    train = [o for o in objs if not is_val(o)]; val = [o for o in objs if is_val(o)][: a.val_objects]
    if not train:
        raise SystemExit(f"no rendered objects under {a.data}")
    log(f"{len(objs)} objects ({len(train)} train / {len(val)} val used); device {device}; backbone {a.backbone}")
    bb_kw = dict((k, v) for k, v in (x.split("=", 1) for x in a.backbone_kw))
    bb = get_backbone(a.backbone, **bb_kw)
    head = GaussianHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    perc = Perceptual().to(device) if a.lpips > 0 else None
    step, best = 0, -1.0
    if a.resume and (out / "ckpt_last.pt").exists():
        ck = torch.load(out / "ckpt_last.pt", map_location=device, weights_only=False)
        head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"]); step, best = ck["step"], ck.get("best", -1.0)
        log(f"resumed at step {step}")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "n/a"
    json.dump(dict(vars(a), commit=commit, n_train=len(train), n_val=len(val), head_params=sum(p.numel() for p in head.parameters())),
              open(out / "config.json", "w"), indent=1)
    mcsv = open(out / "metrics.csv", "a"); mw = csv.writer(mcsv)
    if mcsv.tell() == 0:
        mw.writerow(["step", "loss", "psnr", "n_gauss", "s_per_step", "lr"])
    rng = random.Random(a.seed + step)
    res = None; t_last = time.time(); acc = []

    def validate(tag):
        head.eval(); rows = []
        with torch.no_grad():
            for o in val:
                d = G.load_views(o); V = len(d["paths"]); rr = random.Random(zlib.crc32(o.name.encode()))
                ids = rr.sample(range(V), a.n_in + a.n_tgt)
                pred, gt, n = step_batch(bb, head, G, d, np.array(ids[: a.n_in]), np.array(ids[a.n_in:]), d["res"], device, 1, a.source)
                if pred is None:
                    continue
                rows.append([psnr(pred, gt).mean().item(), ssim(pred, gt).item(), perc(pred, gt).item() if perc else float("nan")])
        head.train()
        m = np.array(rows).mean(0) if rows else np.array([np.nan] * 3)
        with open(out / "val.csv", "a") as f:
            csv.writer(f).writerow([tag, step, *[f"{v:.4f}" for v in m], len(rows)])
        log(f"val@{step}: PSNR {m[0]:.2f}  SSIM {m[1]:.4f}  LPIPS {m[2]:.4f}  ({len(rows)} objects)")
        return m[0]

    while step < a.steps:
        o = rng.choice(train)
        try:
            d = G.load_views(o)
        except Exception as ex:
            log(f"skip {o.name}: {ex}"); continue
        res = d["res"]; V = len(d["paths"])
        ids = rng.sample(range(V), a.n_in + a.n_tgt)
        pred, gt, n = step_batch(bb, head, G, d, np.array(ids[: a.n_in]), np.array(ids[a.n_in:]), res, device, a.reads, a.source)
        if pred is None:
            log(f"skip {o.name}: no valid Gaussians"); continue
        loss = torch.nn.functional.mse_loss(pred, gt)
        if perc is not None:
            loss = loss + a.lpips * perc(pred, gt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step(); step += 1
        acc.append([loss.item(), psnr(pred, gt).mean().item(), n])
        if step % (2 if a.smoke else 50) == 0:
            m = np.array(acc).mean(0); dt = (time.time() - t_last) / len(acc); t_last = time.time(); acc = []
            mw.writerow([step, f"{m[0]:.5f}", f"{m[1]:.3f}", int(m[2]), f"{dt:.3f}", f"{sched.get_last_lr()[0]:.2e}"]); mcsv.flush()
            log(f"step {step}: loss {m[0]:.4f}  psnr {m[1]:.2f}  gaussians {int(m[2]):,}  {dt:.2f} s/step")
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save(dict(head=head.state_dict(), opt=opt.state_dict(), step=step, best=best, args=vars(a)), out / "ckpt_last.pt")
        if step % a.val_every == 0 or step == a.steps:
            v = validate("val")
            if v > best:
                best = v; torch.save(dict(head=head.state_dict(), step=step, args=vars(a), val_psnr=v), out / "ckpt_best.pt")
    log(f"done: {step} steps; best val PSNR {best:.2f}; outputs in {out}")


if __name__ == "__main__":
    main()
