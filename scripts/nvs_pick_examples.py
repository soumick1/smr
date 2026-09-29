#!/usr/bin/env python3
"""Pick GSO objects where the +SMR read helps most, for the qualitative figure.

    python scripts/nvs_pick_examples.py --raw outputs/nvs/vggt/gso_reads1.jsonl --smr outputs/nvs/vggt/gso_reads4.jsonl --top 12
    python scripts/nvs_pick_examples.py ... --rank lpips --min-smr-psnr 20 --max-raw-psnr 22

Joins the two per-object jsonl files (reads 1 = raw, reads 4 = +SMR), ranks by --rank:
  psnr      PSNR gain (smr - raw)                       [default]
  lpips     LPIPS reduction (raw - smr), the more perceptually visible criterion
  combined  mean of the two rank positions
Filters: --min-smr-psnr (the +SMR render must look good), --max-raw-psnr (the raw render must look bad), --min-dpsnr.
Prints the distribution of gains, the top-K table, and a ready --ids line for scripts/nvs_render_objects.py.
"""
import argparse, json, numpy as np

def load(p):
    out = {}
    for l in open(p):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        if "psnr" in r: out[r["id"]] = r
    return out

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--raw", required=True); ap.add_argument("--smr", required=True)
ap.add_argument("--rank", default="psnr", choices=["psnr", "lpips", "combined"]); ap.add_argument("--top", type=int, default=12)
ap.add_argument("--min-smr-psnr", type=float, default=0.0); ap.add_argument("--max-raw-psnr", type=float, default=99.0); ap.add_argument("--min-dpsnr", type=float, default=0.0)
a = ap.parse_args()
A, B = load(a.raw), load(a.smr)
ids = sorted(set(A) & set(B))
rows = []
for i in ids:
    r, s = A[i], B[i]
    rows.append(dict(id=i, raw=r["psnr"], smr=s["psnr"], dpsnr=s["psnr"] - r["psnr"], lraw=r["lpips"], lsmr=s["lpips"], dlpips=r["lpips"] - s["lpips"],
                     sraw=r["ssim"], ssmr=s["ssim"], dssim=s["ssim"] - r["ssim"]))
d = np.array([x["dpsnr"] for x in rows]); dl = np.array([x["dlpips"] for x in rows])
print(f"{len(rows)} objects paired. PSNR gain percentiles (50/90/99/max): {np.percentile(d, 50):.2f} / {np.percentile(d, 90):.2f} / "
      f"{np.percentile(d, 99):.2f} / {d.max():.2f} dB;  LPIPS reduction 50/90/99/max: {np.percentile(dl, 50):.3f} / {np.percentile(dl, 90):.3f} / "
      f"{np.percentile(dl, 99):.3f} / {dl.max():.3f};  objects with gain > 2 dB: {(d > 2).sum()}, > 3 dB: {(d > 3).sum()}")
keep = [x for x in rows if x["smr"] >= a.min_smr_psnr and x["raw"] <= a.max_raw_psnr and x["dpsnr"] >= a.min_dpsnr]
if a.rank == "psnr":
    keep.sort(key=lambda x: -x["dpsnr"])
elif a.rank == "lpips":
    keep.sort(key=lambda x: -x["dlpips"])
else:
    rp = {x["id"]: k for k, x in enumerate(sorted(keep, key=lambda x: -x["dpsnr"]))}
    rl = {x["id"]: k for k, x in enumerate(sorted(keep, key=lambda x: -x["dlpips"]))}
    keep.sort(key=lambda x: rp[x["id"]] + rl[x["id"]])
print(f"\n{'#':>2} {'object':<40} {'raw dB':>7} {'+SMR dB':>8} {'gain':>6} | {'LPIPS raw':>9} {'+SMR':>6} {'red.':>6} | {'SSIM raw':>8} {'+SMR':>6}")
for k, x in enumerate(keep[: a.top]):
    print(f"{k + 1:>2} {x['id'][:40]:<40} {x['raw']:>7.2f} {x['smr']:>8.2f} {x['dpsnr']:>+6.2f} | {x['lraw']:>9.3f} {x['lsmr']:>6.3f} {x['dlpips']:>+6.3f} | {x['sraw']:>8.3f} {x['ssmr']:>6.3f}")
print("\n--ids " + " ".join(x["id"] for x in keep[: a.top]))
