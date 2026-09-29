#!/usr/bin/env python3
"""Render chosen GSO objects with raw (reads 1) and +SMR (reads 4) geometry through the SAME frozen head, save every
input / target / prediction, and report per-target metrics so the figure can show, per object, the target view where
the read helps most.

    python scripts/nvs_render_objects.py --backbone vggt --ckpt outputs/nvs/vggt/ckpt_best.pt --gso ~/data/nvs/gso \
        --ids OBJ1 OBJ2 OBJ3 --out outputs/nvs/vggt/images_pick

Writes <out>/<id>/{input_j.png, gt_t.png, pred_reads1_t.png, pred_reads4_t.png, metrics_reads1.json, metrics_reads4.json,
per_target.json} (the layout scripts/fig_nvs_strip.py expects) and prints, per object, the mean metrics and the target
index with the largest PSNR gain and the largest LPIPS reduction. Mirrors experiments/eval_gso.py exactly (same
predict_geometry, head_input, head and render calls), so the numbers reproduce the jsonl entries.
"""
import argparse, json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--backbone", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--gso", default="~/data/nvs/gso")
ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--reads", nargs="+", type=int, default=[1, 4]); ap.add_argument("--n-in", type=int, default=4)
ap.add_argument("--source", default="auto"); ap.add_argument("--fallback", default="median"); ap.add_argument("--backbone-kw", action="append", default=[])
a = ap.parse_args()
import torch
from PIL import Image
from smr.backbones import get_backbone
from smr.nvs import geometry as G
from smr.nvs.gaussian_head import GaussianHead, Perceptual, psnr, render, ssim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = pathlib.Path(a.gso).expanduser()
bb = get_backbone(a.backbone, **dict(x.split("=", 1) for x in a.backbone_kw))
head = GaussianHead().to(device); ck = torch.load(a.ckpt, map_location=device, weights_only=False); head.load_state_dict(ck["head"]); head.eval()
perc = Perceptual().to(device)
to8 = lambda t: (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
summary = {}
with torch.no_grad():
    for oid in a.ids:
        o = root / oid
        if not (o / "cams.json").exists():
            print(f"{oid}: not found under {root}"); continue
        d = G.load_views(o)
        roles = d["roles"]
        in_ids = np.array([i for i, r in enumerate(roles) if r == "input"][: a.n_in], int)
        tgt_ids = np.array([i for i, r in enumerate(roles) if r == "target"], int)
        if len(in_ids) < a.n_in or len(tgt_ids) == 0:
            n_v = len(roles); in_ids = np.arange(a.n_in); tgt_ids = np.arange(a.n_in, min(n_v, a.n_in + 10))
        sd = out / oid; sd.mkdir(exist_ok=True)
        for j, i in enumerate(in_ids):
            Image.fromarray((d["rgb"][i] * 255).astype("uint8")).save(sd / f"input_{j}.png")
        gt = torch.from_numpy(d["rgb"][tgt_ids]).to(device).permute(0, 3, 1, 2)
        for j in range(gt.shape[0]):
            Image.fromarray(to8(gt[j])).save(sd / f"gt_{j}.png")
        per = {}
        for reads in a.reads:
            geo = G.predict_geometry(bb, [d["paths"][i] for i in in_ids], d["c2w"][in_ids], d["K"], d["res"], reads=reads, source=a.source,
                                     fallback=a.fallback, align=True)
            x, valid = G.head_input(d["rgb"][in_ids], geo["pts"], geo["conf"], d["alpha"][in_ids])
            xt = torch.from_numpy(x).to(device).permute(0, 3, 1, 2)
            g = head(xt, xt[:, 3:6], xt[:, 0:3], torch.from_numpy(valid).to(device))
            pred, _ = render(g, torch.from_numpy(d["c2w"][tgt_ids]).to(device), torch.from_numpy(d["K"]).to(device), d["res"])
            p_t = [float(v) for v in psnr(pred, gt)]
            l_t = [float(perc(pred[j:j + 1], gt[j:j + 1]).item()) for j in range(pred.shape[0])]
            s_t = [float(ssim(pred[j:j + 1], gt[j:j + 1]).item()) for j in range(pred.shape[0])]
            for j in range(pred.shape[0]):
                Image.fromarray(to8(pred[j])).save(sd / f"pred_reads{reads}_{j}.png")
            (sd / f"metrics_reads{reads}.json").write_text(json.dumps(dict(psnr=float(np.mean(p_t)), ssim=float(np.mean(s_t)), lpips=float(np.mean(l_t)),
                                                                            n_gauss=int(g["means"].shape[0]))))
            per[reads] = dict(psnr=p_t, lpips=l_t, ssim=s_t)
        (sd / "per_target.json").write_text(json.dumps(per))
        line = f"{oid}: " + "  ".join(f"reads{r} PSNR {np.mean(per[r]['psnr']):.2f} LPIPS {np.mean(per[r]['lpips']):.3f}" for r in a.reads)
        if len(a.reads) == 2:
            r0, r1 = a.reads
            gp = np.array(per[r1]["psnr"]) - np.array(per[r0]["psnr"]); gl = np.array(per[r0]["lpips"]) - np.array(per[r1]["lpips"])
            line += (f"  | best target by PSNR gain: {int(np.argmax(gp))} ({gp.max():+.2f} dB, {per[r0]['psnr'][int(np.argmax(gp))]:.2f} -> "
                     f"{per[r1]['psnr'][int(np.argmax(gp))]:.2f}); by LPIPS reduction: {int(np.argmax(gl))} ({gl.max():+.3f})")
            summary[oid] = dict(best_psnr_target=int(np.argmax(gp)), best_lpips_target=int(np.argmax(gl)), gain_psnr=[float(v) for v in gp], gain_lpips=[float(v) for v in gl])
        print(line, flush=True)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print("wrote", out / "summary.json")
