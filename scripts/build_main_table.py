#!/usr/bin/env python3
"""Build the merged main table: Table-1 rows (camera pose AUC@30) + DTU + ETH3D columns.

    python scripts/build_main_table.py --main ~/paper/main.tex --matrix outputs/reports/matrix --out paper/tab_main.tex

Camera-pose cells are taken verbatim from the tab:pose block of main.tex (markers and
bold preserved).  DTU cells are 22-scan means of the matrix logs (tags <bb>_raw, <bb>_smr,
<bb>_win_raw, <bb>_win_ca); ETH3D cells are 13-scene means of <bb>_eth.jsonl (sources
single/fused) and <bb>_ethwin_{raw,ca}.jsonl.  Missing cells are '--'; partial coverage is
marked with the count as a superscript.  Row semantics: frozen backbones raw = native pass,
+SMR = the 4-ordering read, +SMR+PGO = the read (no junction graph inside one window);
streaming backbones native = one causal pass, (raw, windowed) = 16-view windows fused by their
median, +SMR = windows + content re-measure, +SMR+PGO = same.
"""
import argparse, collections, json, pathlib, re
import numpy as np

STD22 = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
LABEL2BB = {"DUSt3R": "dust3r", "Fast3R": "fast3r", "MASt3R": "mast3r", "VGGT": "vggt",
            "VGGT-$\\Omega$": "vggt_omega", "$\\pi^3$": "pi3", "StreamVGGT": "streamvggt", "STream3R": "stream3r"}

ap = argparse.ArgumentParser()
ap.add_argument("--main", default="paper/tab_pose_camera.tex", help="file containing the ORIGINAL camera-only tab:pose table (main.tex or paper/tab_pose_camera.tex)")
ap.add_argument("--matrix", default="outputs/reports/matrix")
ap.add_argument("--out", default="paper/tab_main.tex")
a = ap.parse_args()

# ------------------------------------------------------------- camera rows ---
tex = pathlib.Path(a.main).read_text()
blk = re.search(r"\\begin\{tabular\}.*?\\label\{tab:pose\}.*?\\end\{tabular\}|\\label\{tab:pose\}.*?\\begin\{tabular\}(.*?)\\end\{tabular\}", tex, re.S)
if blk is None:
    raise SystemExit("tab:pose tabular not found in main.tex")
body = blk.group(1) if blk.group(1) else blk.group(0)
lines = [l.rstrip() for l in body.splitlines()]
start = next(i for i, l in enumerate(lines) if l.strip().startswith("\\midrule")) + 1
rows = []                                            # (kind, payload)
bb = None; stream = False
for l in lines[start:]:
    s = l.strip()
    if not s or s.startswith("\\bottomrule"):
        continue
    if s.startswith("\\midrule"):
        rows.append(("midrule", None)); continue
    if s.startswith("\\multicolumn{9}"):
        rows.append(("note", s.replace("\\multicolumn{9}", "\\multicolumn{15}"))); continue
    cells = [c.strip() for c in s.rstrip("\\").split("&")]
    label, cam = cells[0], cells[1:9]
    m = re.match(r"(.+?) \((raw|native)\)$", label)
    if m:
        bb = LABEL2BB.get(m.group(1), m.group(1).lower()); stream = m.group(2) == "native"
        kind = "native" if stream else "raw"
    elif "(raw, windowed)" in label:
        kind = "win_raw"
    elif "+PGO" in label:
        kind = "win_pgo" if stream else "pgo"
    elif "\\smr" in label:
        kind = "win_smr" if stream else "smr"
    else:
        kind = "unknown"
    if bb == "vggt_omega" and kind in ("raw", "native"):
        label = label + r"$^{\S}$"                    # released checkpoint: possible benchmark contamination (authors' README)
    rows.append(("data", dict(label=label, bb=bb, kind=kind, cam=cam)))

# --------------------------------------------------------------- dense cells ---
M = pathlib.Path(a.matrix)
dtu = collections.defaultdict(dict)                  # tag -> scan -> [acc, comp, overall]
dtu_status = collections.defaultdict(dict)           # tag -> scan -> "OOM" | "FAILED" (units without a result)
hdr = re.compile(r"== (?P<tag>.+?) scan(?P<scan>\d+) ==")
for fn in M.glob("*_dtu.log"):
    txt = fn.read_text(); hs = list(hdr.finditer(txt))
    for i, h in enumerate(hs):
        block = txt[h.end(): hs[i + 1].start() if i + 1 < len(hs) else len(txt)]
        m = re.search(r"scan\d+: Acc ([\d.]+)  Comp ([\d.]+)  Overall ([\d.]+)", block)
        tag, scan = h.group("tag"), int(h.group("scan"))
        if m:
            dtu[tag][scan] = [float(m.group(i)) for i in (1, 2, 3)]; dtu_status[tag].pop(scan, None)
        elif re.search(r"scan\d+: OOM", block):
            dtu_status[tag].setdefault(scan, "OOM")
        elif re.search(r"scan\d+: (FAILED|MISSING)", block):
            dtu_status[tag].setdefault(scan, "FAILED")
eth = collections.defaultdict(dict)                  # key -> scene -> [acc, comp, overall]
eth_status = collections.defaultdict(dict)
for fn in M.glob("*.jsonl"):
    stem = fn.stem                                   # <bb>_eth | <bb>_ethwin_raw | <bb>_ethwin_ca
    for l in fn.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l); scene = pathlib.Path(r["maps"]).parent.name
        key = f"{stem}:{r['source']}"
        if "umeyama" in r:
            u = r["umeyama"]; eth[key][scene] = [u["acc"], u["comp"], u["overall"]]; eth_status[key].pop(scene, None)
        elif r.get("status"):
            eth_status[key].setdefault(scene, r["status"])

def mean_cells(d, status, n_full, fmt):
    """Means over the finished units; superscript = count when incomplete; OOM when nothing finished
    and at least one unit ran out of memory; -- when nothing is available."""
    if not d:
        return (["OOM"] * 3 if any(v == "OOM" for v in status.values()) else ["--"] * 3), 0
    A = np.array(list(d.values())); n = len(A)
    sup = "" if n >= n_full else f"$^{{\\,{n}}}$"
    return [fmt % v + (sup if i == 2 else "") for i, v in enumerate(A.mean(0))], n

def dtu_cells(tag):
    d = {s: v for s, v in dtu.get(tag, {}).items() if s in STD22}
    return mean_cells(d, dtu_status.get(tag, {}), 22, "%.3f")

def eth_cells(key):
    return mean_cells(eth.get(key, {}), eth_status.get(key, {}), 13, "%.3f")

def cells_for(bb, kind):
    if kind == "raw" or kind == "native":
        return dtu_cells(f"{bb}_raw"), eth_cells(f"{bb}_eth:single")
    if kind in ("smr", "pgo"):
        return dtu_cells(f"{bb}_smr"), eth_cells(f"{bb}_eth:fused")
    if kind == "win_raw":
        return dtu_cells(f"{bb}_win_raw"), eth_cells(f"{bb}_ethwin_raw:fused")
    if kind in ("win_smr", "win_pgo"):
        return dtu_cells(f"{bb}_win_ca"), eth_cells(f"{bb}_ethwin_ca:fused")
    return (["--"] * 3, 0), (["--"] * 3, 0)

def num(c):
    try:
        return float(re.match(r"[\d.]+", c).group(0))
    except Exception:
        return None

# bold the Overall of a +SMR/+PGO row when it improves on the group's raw/windowed-raw row
out_lines = []
group_raw = {}
for kind, payload in rows:
    if kind == "data":
        (dc, _), (ec, _) = cells_for(payload["bb"], payload["kind"])
        payload["dtu"], payload["eth"] = dc, ec
        if payload["kind"] in ("raw", "win_raw"):
            group_raw[(payload["bb"], payload["kind"] == "win_raw")] = (num(dc[2]), num(ec[2]))
for kind, payload in rows:
    if kind == "midrule":
        out_lines.append("\\midrule"); continue
    if kind == "note":
        out_lines.append(payload); continue
    p = payload; dc, ec = list(p["dtu"]), list(p["eth"])
    if p["kind"] in ("smr", "pgo", "win_smr", "win_pgo"):
        ref = group_raw.get((p["bb"], p["kind"].startswith("win")), (None, None))
        for cells, r in ((dc, ref[0]), (ec, ref[1])):
            v = num(cells[2])
            if r is not None and v is not None and v < r - 0.0005:
                cells[2] = "\\textbf{" + cells[2] + "}"
    out_lines.append(" & ".join([p["label"]] + p["cam"] + dc + ec) + " \\\\")

caption = r"""Camera pose, dense MVS and point maps for every backbone, raw and with the memory.
\textbf{Camera pose} (AUC@30$\uparrow$ versus number of input frames $N$): raw = Sim(3)-chained
windows; +\smr{} = causal scaffold memory; +\smr{}+PGO adds the final robust solve. At $N{=}10$
(one window) and, on these data, $N{=}50$ (no revisits yet) the three rows are identical --- the
floor of Sec.~\ref{sec:correct}, measured. $^{\dagger}$DUSt3R/MASt3R length axis on a stated
10-sequence CO3D subset (pairwise-alignment runtime); their RE10K length cells are a 10-clip
subset$^{\ddagger}$. Native and $^{\ddagger}$ windowed cells at $N{=}10$ differ only by sample
(100 vs.\ 10 clips). OOM = exceeds a 46\,GB GPU. RE10K frames are uniform $\le$480p
re-extractions shared by all rows.
\textbf{Dense MVS} (DTU, 22 standard scans, all 49 views, Chamfer mm$\downarrow$) and
\textbf{point maps} (ETH3D, 13 scenes, ten frames, official rendered depth and masks,
Umeyama on per-pixel correspondences, Chamfer m$\downarrow$): one harness for every row.
Official DTU evaluator (masks, 0.2\,mm downsampling); gauge = Sim(3) refinement onto the scan,
chosen by a symmetric fit criterion among the camera alignment, a region-restricted
point-to-plane ICP, and the same from a robust similarity initialisation. Each backbone
contributes its native point map (point head where it has one, depth $\times$ camera
otherwise) with confidence $>2$ (top 68\% for sigmoid confidences). Both benchmarks fit one
window: raw is the single pass; +\smr{} is the memory's in-window read --- four orderings of the
same views written to one scaffold, symmetric re-measure of per-pass scale, consensus (median
on DTU, corroborated pixels only on ETH3D). No junction graph exists inside one window, so
+\smr{}+PGO equals +\smr{}. The read is the identity for order-invariant backbones (DUSt3R and
MASt3R: pairwise + global alignment; $\pi^3$: permutation-equivariant), and it presupposes
commensurable passes: for Fast3R and the causal StreamVGGT/STream3R, orderings yield
different reconstructions and the corroborated read keeps too little (ETH3D). DUSt3R on DTU
uses a sliding-window pair graph (window 5); its complete graph over 49 views exceeds 46\,GB.
$^{\S}$The VGGT-$\Omega$ authors report possible benchmark contamination of the released
checkpoint; ETH3D's training scenes are this benchmark. Streaming backbones: native = one causal
pass; windowed rows use 16-view windows fused by their median, +\smr{} adds the content
re-measure. Published anchors under the authors' own protocols: VGGT DTU 0.389/0.374/0.382
(not reproduced feed-forward by \citet{langendoerfer2026vggt} nor by us), ETH3D 0.873/0.482/0.677.
Superscripts on Overall give the number of scans/scenes when a row is incomplete.
"""

import datetime as _dt
head = f"% built {_dt.datetime.now():%Y-%m-%d %H:%M} by scripts/build_main_table.py from {a.matrix}\n" + r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2.4pt}
\caption{""" + caption.strip() + r"""}
\label{tab:pose}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}l cc cc cc cc ccc ccc@{}}
\toprule
& \multicolumn{8}{c}{Camera pose: AUC@30$\uparrow$} & \multicolumn{3}{c}{Dense MVS: DTU, Chamfer mm$\downarrow$} & \multicolumn{3}{c}{Point maps: ETH3D, Chamfer m$\downarrow$} \\
\cmidrule(lr){2-9}\cmidrule(lr){10-12}\cmidrule(lr){13-15}
& \multicolumn{2}{c}{$N{=}10$} & \multicolumn{2}{c}{$N{=}50$} & \multicolumn{2}{c}{$N{=}100$} & \multicolumn{2}{c}{$N{=}200$} & \multicolumn{3}{c}{49 views, 22 scans} & \multicolumn{3}{c}{10 frames, 13 scenes} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-12}\cmidrule(lr){13-15}
& RE10K & CO3D & RE10K & CO3D & RE10K & CO3D & RE10K & CO3D & Acc & Comp & Overall & Acc & Comp & Overall \\
\midrule
"""
tail = "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(a.out).write_text(head + "\n".join(out_lines) + "\n" + tail)
filled = sum(1 for k, p in rows if k == "data" and p["dtu"][0] not in ("--", "OOM"))
oom = {t: sorted(s for s, v in st.items() if v == "OOM") for t, st in dtu_status.items() if any(v == "OOM" for v in st.values())}
print(f"wrote {a.out}: {sum(1 for k, _ in rows if k == 'data')} rows, {filled} with DTU cells; "
      f"dtu tags {sorted(dtu)}; eth keys {sorted(eth)}" + (f"; OOM units {oom}" if oom else ""))
