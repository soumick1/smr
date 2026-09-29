#!/usr/bin/env python3
"""SD across scans / scenes for the DTU + ETH3D dense table (Table 1 dense columns / tab:pointcloud).

    python scripts/dense_table_sd.py --matrix outputs/reports/matrix

DTU: <matrix>/*_dtu.log with blocks "== <tag> scanNN ==" followed by "scanNN: Acc a  Comp c  Overall o" (mm), tags
<bb>_raw, <bb>_smr, <bb>_win_raw, <bb>_win_ca; the 22 standard evaluation scans only.
ETH3D: <matrix>/<bb>_eth.jsonl (records with "source" single | fused and a "umeyama" dict acc/comp/overall, metres) and
<bb>_ethwin_raw.jsonl / <bb>_ethwin_ca.jsonl.
Prints every tag with n, mean and SD (ddof = 1) for Acc, Comp, Overall, so the means can be matched to the table, then the
SD grid using the mapping frozen: Baseline = raw / eth:single, +SMR = smr / eth:fused; streaming: Baseline = raw /
eth:single (native causal pass), +SMR = win_ca / ethwin_ca:fused. Change --smr-tag-stream if the table used another row.
"""
import argparse, collections, json, pathlib, re
import numpy as np

STD22 = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
ORDER = ["dust3r", "fast3r", "mast3r", "vggt", "vggt_omega", "pi3", "streamvggt", "stream3r"]
STREAM = {"streamvggt", "stream3r"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--matrix", default="outputs/reports/matrix")
ap.add_argument("--smr-tag-stream", default="win_ca", help="DTU tag suffix for the streaming +SMR row (win_ca | win_raw | smr)")
ap.add_argument("--eth-smr-stream", default="ethwin_ca:fused", help="ETH key for the streaming +SMR row")
a = ap.parse_args()
M = pathlib.Path(a.matrix)

dtu = collections.defaultdict(dict)
hdr = re.compile(r"== (?P<tag>.+?) scan(?P<scan>\d+) ==")
for fn in M.glob("*_dtu.log"):
    txt = fn.read_text(); hs = list(hdr.finditer(txt))
    for i, h in enumerate(hs):
        block = txt[h.end(): hs[i + 1].start() if i + 1 < len(hs) else len(txt)]
        m = re.search(r"scan\d+: Acc ([\d.]+)  Comp ([\d.]+)  Overall ([\d.]+)", block)
        if m and int(h.group("scan")) in STD22:
            dtu[h.group("tag")][int(h.group("scan"))] = [float(m.group(k)) for k in (1, 2, 3)]
eth = collections.defaultdict(dict)
for fn in M.glob("*.jsonl"):
    for l in fn.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if "umeyama" in r:
            u = r["umeyama"]; scene = pathlib.Path(r["maps"]).parent.name
            eth[f"{fn.stem}:{r['source']}"][scene] = [u["acc"], u["comp"], u["overall"]]


def summarise(d, unit):
    if not d:
        return None
    A = np.array(list(d.values()))
    return dict(n=len(A), mean=A.mean(0), sd=A.std(0, ddof=1) if len(A) > 1 else np.full(3, np.nan))


print("== DTU tags (22 standard scans; mm) ==")
for tag in sorted(dtu):
    s = summarise(dtu[tag], "mm")
    print(f"  {tag:<22} n={s['n']:<3} mean {s['mean'][0]:.3f} {s['mean'][1]:.3f} {s['mean'][2]:.3f}   sd {s['sd'][0]:.3f} {s['sd'][1]:.3f} {s['sd'][2]:.3f}")
print("== ETH3D keys (13 scenes; m) ==")
for key in sorted(eth):
    s = summarise(eth[key], "m")
    print(f"  {key:<28} n={s['n']:<3} mean {s['mean'][0]:.3f} {s['mean'][1]:.3f} {s['mean'][2]:.3f}   sd {s['sd'][0]:.3f} {s['sd'][1]:.3f} {s['sd'][2]:.3f}")


def cells(d):
    s = summarise(d, "")
    return ["--"] * 3 if s is None else [f"{v:.3f}" for v in s["sd"]]


print("\n== SD grid (Acc, Comp, Overall) in the table's layout ==")
print(f"{'model':<11} {'row':<9} {'DTU Acc':>8} {'Comp':>8} {'Overall':>8}   {'ETH Acc':>8} {'Comp':>8} {'Overall':>8}")
for bb in ORDER:
    stream = bb in STREAM
    rows = [("Baseline", dtu.get(f"{bb}_raw", {}), eth.get(f"{bb}_eth:single", {})),
            ("+SMR", dtu.get(f"{bb}_{a.smr_tag_stream if stream else 'smr'}", {}), eth.get(f"{bb}_{a.eth_smr_stream}" if stream else f"{bb}_eth:fused", {}))]
    for name, d1, d2 in rows:
        c1, c2 = cells(d1), cells(d2)
        print(f"{bb:<11} {name:<9} {c1[0]:>8} {c1[1]:>8} {c1[2]:>8}   {c2[0]:>8} {c2[1]:>8} {c2[2]:>8}"
              f"   (n={len(d1)}/{len(d2)})")