#!/usr/bin/env python3
"""Evaluate a trained Gaussian head on our GSO renders (LVSM protocol: inputs = views 0-3, targets = 4-13).

    python experiments/eval_gso.py --backbone vggt --ckpt outputs/nvs/vggt/ckpt_best.pt --gso ~/data/nvs/gso \
        --reads 1 --json outputs/nvs/vggt/gso_raw.jsonl
    python experiments/eval_gso.py ... --reads 4 --json outputs/nvs/vggt/gso_read.jsonl      # +SMR (in-window read)

Per object: geometry from the frozen backbone (reads=1: raw pass; reads>1: the memory's read --
K orderings, symmetric re-measure, consensus), the SAME head, the 10 targets rendered and scored
(PSNR / SSIM / LPIPS, full image on white).  Appends one json line per object (resume-safe: objects
already in the file are skipped) and prints the mean row.  `--summary a.jsonl b.jsonl` prints rows only.
"""
import argparse, json, pathlib, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def summarize(paths):
    for p in paths:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        ok = [r for r in rows if "psnr" in r]
        if not ok:
            print(f"{p}: no results"); continue
        m = np.array([[r["psnr"], r["ssim"], r["lpips"]] for r in ok]).mean(0)
        print(f"{pathlib.Path(p).stem:<24} n={len(ok):4d}  PSNR {m[0]:6.2f}  SSIM {m[1]:.4f}  LPIPS {m[2]:.4f}"
              + (f"  ({len(rows) - len(ok)} failed)" if len(rows) != len(ok) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", nargs="*", default=None)
    ap.add_argument("--backbone"); ap.add_argument("--ckpt"); ap.add_argument("--gso", default="~/data/nvs/gso")
    ap.add_argument("--reads", type=int, default=1); ap.add_argument("--source", default="auto")
    ap.add_argument("--json", default=None); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-in", type=int, default=4)
    ap.add_argument("--backbone-kw", action="append", default=[])
    a = ap.parse_args()
    if a.summary is not None:
        return summarize(a.summary)
    import torch
    from smr.backbones import get_backbone
    from smr.nvs import geometry as G
    from smr.nvs.gaussian_head import GaussianHead, Perceptual, psnr, render, ssim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = pathlib.Path(a.gso).expanduser()
    objs = sorted(p.parent for p in root.glob("*/cams.json"))
    excl = set((root / "exclude.txt").read_text().split()) if (root / "exclude.txt").exists() else set()
    objs = [o for o in objs if not (o / "FAILED").exists() and o.name not in excl]
    if a.limit:
        objs = objs[: a.limit]
    out = pathlib.Path(a.json or f"outputs/nvs/{a.backbone}/gso_reads{a.reads}.jsonl"); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out) if l.strip()} if out.exists() else set()
    crashed_f = out.with_suffix(".crashed"); inflight_f = out.with_suffix(".inflight")
    crashed = set(crashed_f.read_text().split()) if crashed_f.exists() else set()
    bb = get_backbone(a.backbone, **dict(x.split("=", 1) for x in a.backbone_kw))
    head = GaussianHead().to(device); ck = torch.load(a.ckpt, map_location=device, weights_only=False); head.load_state_dict(ck["head"]); head.eval()
    perc = Perceptual().to(device)
    print(f"{len(objs)} GSO objects ({len(done)} done); backbone {a.backbone}, reads {a.reads}, head from step {ck.get('step')}", flush=True)
    t0 = time.time()
    with torch.no_grad(), open(out, "a") as f:
        for k, o in enumerate(objs):
            if o.name in done:
                continue
            if o.name in crashed:                                   # killed the process before (see the launcher)
                f.write(json.dumps(dict(id=o.name, reads=a.reads, error="process crashed on this object")) + "\n"); f.flush(); continue
            inflight_f.write_text(o.name)                           # who was in flight if we die without a Python exception
            try:
                d = G.load_views(o)
                roles = d["roles"]
                in_ids = np.array([i for i, r in enumerate(roles) if r == "input"][: a.n_in])
                tgt_ids = np.array([i for i, r in enumerate(roles) if r == "target"])
                geo = G.predict_geometry(bb, [d["paths"][i] for i in in_ids], d["c2w"][in_ids], d["K"], d["res"], reads=a.reads, source=a.source)
                x, valid = G.head_input(d["rgb"][in_ids], geo["pts"], geo["conf"], d["alpha"][in_ids])
                xt = torch.from_numpy(x).to(device).permute(0, 3, 1, 2)
                g = head(xt, xt[:, 3:6], xt[:, 0:3], torch.from_numpy(valid).to(device))
                if g["means"].shape[0] == 0:
                    raise RuntimeError("no valid Gaussians (all pixels invalid after the read)")
                pred, _ = render(g, torch.from_numpy(d["c2w"][tgt_ids]).to(device), torch.from_numpy(d["K"]).to(device), d["res"])
                gt = torch.from_numpy(d["rgb"][tgt_ids]).to(device).permute(0, 3, 1, 2)
                rec = dict(id=o.name, reads=a.reads, psnr=psnr(pred, gt).mean().item(), ssim=ssim(pred, gt).item(),
                           lpips=perc(pred, gt).item(), n_gauss=int(g["means"].shape[0]), scales=[float(s) for s in geo["scales"]])
            except Exception as ex:
                rec = dict(id=o.name, reads=a.reads, error=f"{type(ex).__name__}: {ex}")
            f.write(json.dumps(rec) + "\n"); f.flush()
            inflight_f.unlink(missing_ok=True)
            if (k + 1) % 25 == 0:
                print(f"  {k+1}/{len(objs)} ({time.time()-t0:.0f} s)", flush=True)
    summarize([str(out)])


if __name__ == "__main__":
    main()
