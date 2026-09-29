#!/usr/bin/env python3
"""Closure reliability from pilot_a reports (v178, plan Task 5).

    python scripts/closure_reliability.py --json outputs2/reports/pilotA_7scenes_*_seq01_vggt_omega_s5_c32.json \
        --gt-dir data/gt --md outputs/rel_7scenes_seq01.md --csv outputs/rel_7scenes_seq01.csv
    python scripts/closure_reliability.py --glob 'outputs/7scenes_test/*_vggt_omega_template.json' --gt-dir data/gt --md R.md
    python scripts/closure_reliability.py --glob 'outputs/ablate/gate/7scenes/default/*.json' --gt-dir data/gt \
        --est-dir outputs/ablate/gate/7scenes/default --md R.md          # + per-closure usefulness from --save-est npz

Three independent correctness definitions for an accepted closure (k = query window, its verified historical views v):
  strict   some v is a GT revisit of a new frame of k: GT centres within `dist` (default 10 % of the GT bounding-box
           diagonal) and GT orientations within `angle` (default 30 deg). This is "same place"; it is stricter than
           what a valid loop closure needs.
  2x       the same at twice the distance.
  covis    a co-visibility proxy: centres within `covis_frac` of the diagonal (default 0.30) and viewing directions
           within `covis_angle` (default 60 deg). Two cameras looking at the same wall from 1 m apart are a valid
           closure; the strict test calls that false.
  useful   (needs the --save-est npz) the closure's EFFECT on the placement of the two windows it connects: mean over
           frame pairs of || (p_i - p_j)_est - (p_i - p_j)_gt || after each trajectory's own Sim(3) alignment to GT
           (metres in metric datasets), raw chain vs the row. useful = falls by > 10 %, harmful = rises by > 10 %.
           The angular pair error (AUC quantity) is reported alongside but is dominated by within-window noise.

The eligible history for window k is what the stitcher's proposal stage allows: stored views whose owner window is
< k-1 and which lie before the recent window (2W keyframes). GT files are resolved from the report FILE NAME first
(the `seqNN` token must match: chess_seq03 never maps to 7scenes_chess_seq01.npz) and from the report's scene field
second; the resolved file is printed for every report. Reads v167 reports (sites + loop per window) and v173+
reports (adds events[].candidates for recall@1/@5).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smr.eval.trajectory import rotation_angle_deg  # noqa: E402
from smr.stitch.chunks import keyframe_indices, make_chunks  # noqa: E402

SITE_ROT, SITE_DIR, EXTENT = 10.0, 25.0, 3.0            # AnchoredStitcher defaults (tests/test_guarantees.py::F)


# ------------------------------------------------------------------ inputs
def _norm(s):
    s = re.sub(r"seq[-_ ]?0*(\d+)", r"seq\1", str(s).lower())        # seq03 == seq-03 == seq_3
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _rebuild_ok(report, gt_path):
    """A GT file is admissible only if the report's keyframes/windows rebuild from it."""
    try:
        gt_and_chunks(report, gt_path)
        return True
    except Exception:  # noqa: BLE001
        return False


def find_gt(report, report_path, a, debug=False):
    """Resolve the GT npz. Order: --gt; --gt-pattern (with {stem} = report file stem, {scene} = report scene field);
    else search --gt-dir. Scoring: tokens of the GT stem that appear in the report stem count +1 each, tokens that do
    not appear count -1 (so `7scenes_chess_all` loses to `7scenes_chess_seq01` for a report called `chess` only if
    validation separates them, and `_all`/`_multi`/`_s0106` files are excluded unless the report names them); a `seqNN`
    token in the report name must match; ties are broken by validation (the report's keyframes and windows must rebuild
    from the file) and then by a directory whose name contains the report's dataset."""
    if a.gt:
        return pathlib.Path(a.gt)
    if not a.gt_dir:
        return None
    stem_raw = pathlib.Path(report_path).stem
    stem = _norm(stem_raw)
    scene = _norm(report.get("scene", ""))
    dataset = _norm(report.get("dataset", ""))
    if getattr(a, "gt_pattern", None):
        for pat in str(a.gt_pattern).split(","):
            pat = pat.strip()
            if "=" in pat:                                   # dataset-prefixed: 7scenes=7scenes_{stem}_seq01.npz
                ds, pat = pat.split("=", 1)
                ALIAS = {"7scenes": ("7scenes", "sevenscenes"), "sevenscenes": ("7scenes", "sevenscenes"), "co3d": ("co3d",)}
                names = ALIAS.get(_norm(ds), (_norm(ds),))
                if not any(n in dataset or n in stem for n in names):
                    continue
            cand = pathlib.Path(a.gt_dir) / pat.format(stem=stem_raw, scene=report.get("scene", ""))
            if cand.exists():
                return cand
        if not getattr(find_gt, "_warned", False):
            print("  --gt-pattern did not match some reports; falling back to search for those (message shown once)")
            find_gt._warned = True
    files = [pathlib.Path(p) for p in glob.glob(str(pathlib.Path(a.gt_dir) / "**" / "*.npz"), recursive=True)]
    seq = re.search(r"(seq\d+)", stem)
    SKIP = ("7scenes", "sevenscenes", "co3d", "tum", "gt", "full", "poses")
    MULTI = ("all", "multi", "sessions", "concat")
    scored = []
    for f in files:
        fs = _norm(f.stem)
        if seq and seq.group(1) not in fs:
            continue
        toks = [t for t in fs.split("_") if t not in SKIP]
        if any(t in MULTI or re.fullmatch(r"s\d{4,}", t) for t in toks) and not any(t in stem for t in toks if t in MULTI or re.fullmatch(r"s\d{4,}", t)):
            continue                                                    # multi-session files only when named
        hit = sum(1 for t in toks if t in stem)
        miss = sum(1 for t in toks if t not in stem)
        score = 3 * hit - miss + (2 if scene and scene in fs else 0) + (1 if fs in stem else 0)
        if hit:
            scored.append([score, f])
    if not scored:
        if debug:
            print(f"  gt: no candidate for {stem} (scene={scene})")
        return None
    scored.sort(key=lambda t: -t[0])
    top = [f for sc, f in scored if sc == scored[0][0]]
    if len(top) > 1:
        ok = [f for f in top if _rebuild_ok(report, f)]
        if ok:
            top = ok
    if len(top) > 1 and dataset:
        ds = [f for f in top if dataset in _norm(f.parent.name)]
        if ds:
            top = ds
    if len(top) > 1:
        names = {f.name for f in top}
        if len(names) > 1:
            print(f"  WARNING ambiguous GT for {stem}: {[f.name for f in top[:3]]}; using {top[0].name}")
    if debug:
        print(f"  gt candidates for {stem}: {[(sc, f.name) for sc, f in scored[:4]]} -> {top[0].name}")
    return top[0]


def gt_and_chunks(report, gt_path, est_path=None):
    """(gt poses at the keyframes, chunks, key), from a --save-est npz when
    given, else rebuilt from the report's sampling parameters."""
    npz = np.load(gt_path, allow_pickle=True)
    poses = np.asarray(npz["poses"], float)
    if est_path is not None and pathlib.Path(est_path).exists():
        z = np.load(est_path, allow_pickle=True)
        key = [int(k) for k in z["key"]]
        chunks = [[int(g) for g in np.asarray(c).ravel() if g >= 0] for c in z["chunks"]]
        return poses[key], chunks, key
    key = keyframe_indices(len(poses), int(report["keyframe_stride"]))[: int(report["keyframes"])]
    if len(key) != int(report["keyframes"]):
        raise ValueError(f"cannot rebuild keyframes: {len(key)} vs {report['keyframes']}")
    sessions = None
    if "frame_ids" in npz:
        fid = np.asarray(npz["frame_ids"])
        if np.issubdtype(fid.dtype, np.integer) and fid.max() >= 100000:
            sessions = (fid[key] // 100000).tolist()
    if sessions is not None and len(set(sessions)) > 1:
        from smr.stitch.chunks import make_session_chunks
        chunks = make_session_chunks(sessions, int(report["chunk"]), int(report["overlap"]))
    else:
        chunks = make_chunks(len(key), int(report["chunk"]), int(report["overlap"]))
    if len(chunks) != int(report["n_chunks"]):
        raise ValueError(f"rebuilt {len(chunks)} windows, report has {report['n_chunks']}")
    return poses[key], chunks, key


def owners_and_news(chunks):
    owner, seen, news = {}, set(), []
    for k, idx in enumerate(chunks):
        new = [g for g in idx if g not in seen]
        for g in new:
            owner[g] = k
        seen |= set(idx)
        news.append(new)
    return owner, news


def revisit_truth(gt, chunks, dist=None, angle=30.0, covis_frac=0.30, covis_angle=60.0):
    """Per window k: eligible historical views that are GT revisits of k's new
    frames under the strict / 2x / co-visibility definitions."""
    c = gt[:, :3, 3]
    diag = float(np.linalg.norm(np.ptp(c, axis=0))) + 1e-12
    if dist is None:
        dist = 0.10 * diag
    covis_d = covis_frac * diag
    owner, news = owners_and_news(chunks)
    W = max(len(cc) for cc in chunks)
    fwd = gt[:, :3, 2]                                  # camera viewing direction (OpenCV c2w: +z)
    truth = {}
    for k in range(1, len(chunks)):
        new = news[k]
        if not new:
            truth[k] = (set(), set(), set())
            continue
        lo = min(new) - 2 * W
        elig = [g for g in owner if owner[g] < k - 1 and g < lo]
        s1, s2, s3 = set(), set(), set()
        for j in elig:
            for i in new:
                dd = np.linalg.norm(c[i] - c[j])
                if dd >= 2 * dist and dd >= covis_d:
                    continue
                ra = rotation_angle_deg(gt[i, :3, :3].T @ gt[j, :3, :3])
                va = float(np.degrees(np.arccos(np.clip(fwd[i] @ fwd[j], -1, 1))))
                if ra < angle and dd < dist:
                    s1.add(j)
                if ra < angle and dd < 2 * dist:
                    s2.add(j)
                if va < covis_angle and dd < covis_d:
                    s3.add(j)
        truth[k] = (s1, s2, s3)
    return truth, owner, dist


def pair_error(est, gt, Fa, Fb, max_frames=12):
    """Mean over frame pairs (i in Fa, j in Fb) of max(relative-rotation error,
    translation-direction error) in degrees, est vs gt (the AUC protocol's quantity)."""
    Fa = Fa[:: max(1, len(Fa) // max_frames)][:max_frames]
    Fb = Fb[:: max(1, len(Fb) // max_frames)][:max_frames]
    errs = []
    for i in Fa:
        for j in Fb:
            Re = est[i, :3, :3].T @ est[j, :3, :3]
            Rg = gt[i, :3, :3].T @ gt[j, :3, :3]
            rot = rotation_angle_deg(Re.T @ Rg)
            te = est[i, :3, :3].T @ (est[j, :3, 3] - est[i, :3, 3])
            tg = gt[i, :3, :3].T @ (gt[j, :3, 3] - gt[i, :3, 3])
            ne, ng = np.linalg.norm(te), np.linalg.norm(tg)
            dirr = 0.0 if (ne < 1e-9 or ng < 1e-9) else float(np.degrees(np.arccos(np.clip(te @ tg / (ne * ng), -1, 1))))
            errs.append(max(rot, dirr))
    return float(np.mean(errs)) if errs else float("nan")


def rel_pos_error(est_aligned, gt, Fa, Fb, max_frames=12):
    """Mean over frame pairs of || (p_i - p_j)_est - (p_i - p_j)_gt || with est already Sim(3)-aligned to GT (metres in
    metric datasets): the relative PLACEMENT error of the two windows, which is what a loop closure is meant to fix and
    what ATE measures. Insensitive to the within-window angular noise that dominates pair_error."""
    Fa = Fa[:: max(1, len(Fa) // max_frames)][:max_frames]
    Fb = Fb[:: max(1, len(Fb) // max_frames)][:max_frames]
    pe, pg = est_aligned[:, :3, 3], gt[:, :3, 3]
    d = [np.linalg.norm((pe[i] - pe[j]) - (pg[i] - pg[j])) for i in Fa for j in Fb]
    return float(np.mean(d)) if d else float("nan")


# ---------------------------------------------------------------- analysis
def cand_ids(e):
    cs = e.get("candidates")
    if cs is None:
        return None
    out = []
    for x in cs:
        v = x.get("view", x.get("j", x.get("id"))) if isinstance(x, dict) else (x[0] if isinstance(x, (list, tuple)) else x)
        if v is not None:
            out.append(int(v))
    return out


def analyse(report, truth=None, row="smr", est=None, gt=None, owner=None):
    """Per-window records. est: dict row -> (N,4,4) trajectories incl. 'chained' (from --save-est) for usefulness."""
    ev_all = report.get("events", {})
    ev = ev_all.get(row) or next(iter(ev_all.values()), [])
    recs = []
    frames_of = {}
    aligned = {}
    if est is not None and gt is not None:
        from smr.eval.trajectory import align_to_gt
        for name in ("chained", row):
            if name in est and len(est[name]) == len(gt):
                aligned[name] = align_to_gt(est[name], gt, with_scale=True)[0]
    if owner is not None:
        for g, o in owner.items():
            frames_of.setdefault(o, []).append(g)
        for o in frames_of:
            frames_of[o].sort()
    for e in ev:
        k = int(e["chunk"])
        if k == 0:
            continue
        sites = e.get("sites") or []
        loop = e.get("loop")
        r = dict(chunk=k, n_new=e.get("n_new"), n_sites=len(sites), n_ok=sum(1 for s in sites if s["ok"]),
                 fail_rot=sum(1 for s in sites if not s["ok"] and s.get("rot_deg", 0) >= SITE_ROT),
                 fail_dir=sum(1 for s in sites if not s["ok"] and s.get("dir_deg", 0) >= SITE_DIR),
                 fail_ext=sum(1 for s in sites if not s["ok"] and s.get("extent", 0) >= EXTENT),
                 max_cos=max([s.get("cos", 0.0) for s in sites], default=None),
                 has_loop=bool(loop), accepted=bool(loop and loop.get("accepted")),
                 revoked=bool(loop and "revoked_at" in loop), reason=(loop or {}).get("reason"),
                 n_sites_loop=(loop or {}).get("n_sites"), D_rot=(loop or {}).get("D_rot_deg"),
                 D_pos=(loop or {}).get("D_pos_rel"), D_ls=(loop or {}).get("D_logscale"),
                 b_rot=((loop or {}).get("budget") or {}).get("rot"), b_pos=((loop or {}).get("budget") or {}).get("pos"),
                 n_stretch=(loop or {}).get("n_stretch"), baseline_rel=(loop or {}).get("baseline_rel"),
                 scale_from_anchors=(loop or {}).get("scale_from_anchors"), anchor_chunk=(loop or {}).get("anchor_chunk"),
                 relocalisation=(loop or {}).get("relocalisation"), anchored_pass=len(sites) > 0,
                 anch_rot=((e.get("anchored_distortion") or {}).get("rot_deg")),
                 anch_pos=((e.get("anchored_distortion") or {}).get("pos_rel")),
                 anch_used=((e.get("anchored_distortion") or {}).get("used")))
        if truth is not None and k in truth:
            t1, t2, t3 = truth[k]
            views = [int(s["view"]) for s in sites]
            ok_views = [int(s["view"]) for s in sites if s["ok"]]
            cands = cand_ids(e)
            r.update(gt_revisit=len(t1) > 0, gt_revisit_2x=len(t2) > 0, gt_revisit_covis=len(t3) > 0,
                     site_hit=any(v in t1 for v in views), site_hit_2x=any(v in t2 for v in views),
                     site_hit_covis=any(v in t3 for v in views),
                     ok_true=sum(1 for v in ok_views if v in t1), ok_true_2x=sum(1 for v in ok_views if v in t2),
                     ok_true_covis=sum(1 for v in ok_views if v in t3),
                     closure_true=(r["accepted"] and any(v in t1 for v in ok_views)),
                     closure_true_2x=(r["accepted"] and any(v in t2 for v in ok_views)),
                     closure_true_covis=(r["accepted"] and any(v in t3 for v in ok_views)),
                     recall1=(None if cands is None else (len(cands) > 0 and cands[0] in t1)),
                     recall5=(None if cands is None else any(v in t1 for v in cands[:5])),
                     n_cands=(None if cands is None else len(cands)))
        if aligned and r["accepted"] and r["anchor_chunk"] is not None and "chained" in aligned and row in aligned:
            Fk, Fo = frames_of.get(k, []), frames_of.get(int(r["anchor_chunk"]), [])
            if len(Fk) >= 2 and len(Fo) >= 2:
                pb, pa = rel_pos_error(aligned["chained"], gt, Fk, Fo), rel_pos_error(aligned[row], gt, Fk, Fo)
                ab, aa = pair_error(est["chained"], gt, Fk, Fo), pair_error(est[row], gt, Fk, Fo)
                r.update(pos_before=pb, pos_after=pa, ang_before=ab, ang_after=aa,
                         useful=(pa < 0.9 * pb), harmful=(pa > 1.1 * pb))
        recs.append(r)
    return recs


def summarise(recs, has_gt):
    n = len(recs)
    acc = [r for r in recs if r["accepted"]]
    prop = [r for r in recs if r["n_sites"] > 0]
    loops = [r for r in recs if r["has_loop"]]
    S = dict(windows=n, windows_with_proposals=len(prop), sites_proposed=sum(r["n_sites"] for r in recs),
             sites_verified=sum(r["n_ok"] for r in recs), site_fail_rot=sum(r["fail_rot"] for r in recs),
             site_fail_dir=sum(r["fail_dir"] for r in recs), site_fail_extent=sum(r["fail_ext"] for r in recs),
             closures_tested=len(loops), accepted=len(acc), revoked=sum(1 for r in recs if r["revoked"]),
             rejected_rot=sum(1 for r in loops if not r["accepted"] and r["reason"] == "rot"),
             rejected_pos=sum(1 for r in loops if not r["accepted"] and r["reason"] == "pos"),
             rejected_scale=sum(1 for r in loops if not r["accepted"] and r["reason"] == "scale"),
             accepted_two_site=sum(1 for r in acc if (r["n_sites_loop"] or 0) >= 2),
             accepted_scale_measured=sum(1 for r in acc if r["scale_from_anchors"]),
             anchored_passes_attempted=sum(1 for r in recs if r["anchored_pass"]),
             D_pos_rel_median=(float(np.median([r["D_pos"] for r in acc])) if acc else None),
             D_rot_deg_median=(float(np.median([r["D_rot"] for r in acc])) if acc else None),
             n_stretch_median=(float(np.median([r["n_stretch"] for r in acc])) if acc else None))
    if has_gt:
        gtw = [r for r in recs if r["gt_revisit"]]
        gtc = [r for r in recs if r["gt_revisit_covis"]]
        S.update(windows_with_gt_revisit=len(gtw), windows_with_covis_revisit=len(gtc),
                 site_recall=(np.mean([r["site_hit"] for r in gtw]) if gtw else None),
                 site_recall_2x=(np.mean([r["site_hit_2x"] for r in gtw]) if gtw else None),
                 site_recall_covis=(np.mean([r["site_hit_covis"] for r in gtc]) if gtc else None),
                 closure_recall=(np.mean([r["accepted"] for r in gtw]) if gtw else None),
                 closure_recall_covis=(np.mean([r["accepted"] for r in gtc]) if gtc else None),
                 closure_precision=(np.mean([r["closure_true"] for r in acc]) if acc else None),
                 closure_precision_2x=(np.mean([r["closure_true_2x"] for r in acc]) if acc else None),
                 closure_precision_covis=(np.mean([r["closure_true_covis"] for r in acc]) if acc else None),
                 verified_site_precision=(sum(r["ok_true"] for r in recs) / max(1, sum(r["n_ok"] for r in recs))),
                 verified_site_precision_covis=(sum(r["ok_true_covis"] for r in recs) / max(1, sum(r["n_ok"] for r in recs))),
                 false_closures_2x=sum(1 for r in acc if not r["closure_true_2x"]),
                 false_closures_covis=sum(1 for r in acc if not r["closure_true_covis"]),
                 missed_no_proposal=sum(1 for r in gtw if r["n_sites"] == 0),
                 missed_no_verified=sum(1 for r in gtw if r["n_sites"] > 0 and r["n_ok"] == 0),
                 missed_budget=sum(1 for r in gtw if r["has_loop"] and not r["accepted"]),
                 missed_other=sum(1 for r in gtw if r["n_ok"] > 0 and not r["has_loop"]))
        c1 = [r["recall1"] for r in gtw if r.get("recall1") is not None]
        c5 = [r["recall5"] for r in gtw if r.get("recall5") is not None]
        S.update(recall_at_1=(float(np.mean(c1)) if c1 else None), recall_at_5=(float(np.mean(c5)) if c5 else None))
    ad = [r["anch_rot"] for r in recs if r.get("anch_rot") is not None]
    if ad:
        S.update(anchored_distortion_rot_median=float(np.median(ad)), anchored_distortion_rot_max=float(np.max(ad)),
                 anchored_distortion_rot_p90=float(np.percentile(ad, 90)),
                 anchored_distortion_pos_median=float(np.median([r["anch_pos"] for r in recs if r.get("anch_pos") is not None])),
                 windows_reverted_to_plain=sum(1 for r in recs if r.get("anch_used") == "plain"))
    us = [r for r in acc if "useful" in r]
    if us:
        S.update(closures_scored=len(us), useful=sum(1 for r in us if r["useful"]), harmful=sum(1 for r in us if r["harmful"]),
                 pos_before_mean=float(np.mean([r["pos_before"] for r in us])), pos_after_mean=float(np.mean([r["pos_after"] for r in us])),
                 ang_before_mean=float(np.mean([r["ang_before"] for r in us])), ang_after_mean=float(np.mean([r["ang_after"] for r in us])),
                 useful_among_false_covis=(sum(1 for r in us if r["useful"] and not r.get("closure_true_covis", True)) if has_gt else None),
                 harmful_among_true_covis=(sum(1 for r in us if r["harmful"] and r.get("closure_true_covis", False)) if has_gt else None))
    return S


def breakdown(recs):
    """Useful / harmful / neutral rates of accepted, scored closures by the features the stitcher records:
    what separates a harmful closure from a useful one (plan Task 5, failure stratification)."""
    acc = [r for r in recs if r["accepted"] and "useful" in r]
    if not acc:
        return "no scored closures"
    def bucket(r):
        yield "sites=" + ("2" if (r["n_sites_loop"] or 0) >= 2 else "1")
        yield "scale=" + ("measured" if r["scale_from_anchors"] else "fixed")
        yield "span=" + ("1" if (r["n_stretch"] or 0) <= 1 else ("2-3" if (r["n_stretch"] or 0) <= 3 else ">=4"))
        d = r["D_pos"] or 0.0
        yield "D_pos=" + ("<0.05" if d < 0.05 else ("0.05-0.2" if d < 0.2 else ">=0.2"))
        c = r["max_cos"] or 0.0
        yield "cos=" + ("<0.6" if c < 0.6 else ("0.6-0.8" if c < 0.8 else ">=0.8"))
        if "closure_true_covis" in r:
            yield "covis=" + ("true" if r["closure_true_covis"] else "false")
        yield "before=" + ("<0.03" if r["pos_before"] < 0.03 else ("0.03-0.1" if r["pos_before"] < 0.1 else ">=0.1"))
    tab = {}
    for r in acc:
        for b in bucket(r):
            t = tab.setdefault(b, [0, 0, 0])
            t[0 if r["useful"] else (1 if r["harmful"] else 2)] += 1
    lines = [f"{'feature':<18} {'n':>4} {'useful':>7} {'harmful':>8} {'neutral':>8}"]
    for b in sorted(tab):
        u, h, m = tab[b]; n = u + h + m
        lines.append(f"{b:<18} {n:>4} {u / n:>7.2f} {h / n:>8.2f} {m / n:>8.2f}")
    return "\n".join(lines)


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (float, np.floating)):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", nargs="*", default=[]); ap.add_argument("--glob", default=None)
    ap.add_argument("--gt", default=None); ap.add_argument("--gt-dir", default=None); ap.add_argument("--debug-gt", action="store_true")
    ap.add_argument("--gt-pattern", default=None, help="e.g. '7scenes_{stem}_seq01.npz' or 'co3d_full/co3d_{stem}.npz' ({stem}=report file stem, {scene}=scene field)")
    ap.add_argument("--est", default=None, help="--save-est npz of a single report")
    ap.add_argument("--est-dir", default=None, help="directory holding <report stem>.npz from --save-est (usefulness)")
    ap.add_argument("--row", default="smr")
    ap.add_argument("--dist", type=float, default=None); ap.add_argument("--angle", type=float, default=30.0)
    ap.add_argument("--covis-frac", type=float, default=0.30); ap.add_argument("--covis-angle", type=float, default=60.0)
    ap.add_argument("--md", default=None); ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    paths = [pathlib.Path(p) for p in a.json] + ([pathlib.Path(p) for p in sorted(glob.glob(a.glob))] if a.glob else [])
    if not paths:
        sys.exit("no reports")
    all_recs, per = [], []
    for p in paths:
        rep = json.load(open(p))
        gtp = find_gt(rep, p, a, a.debug_gt)
        est_path = a.est if a.est else (pathlib.Path(a.est_dir) / (p.stem + ".npz") if a.est_dir else None)
        truth, est, gt, owner = None, None, None, None
        if gtp is not None:
            try:
                gt, chunks, key = gt_and_chunks(rep, gtp, est_path)
                truth, owner, dist = revisit_truth(gt, chunks, a.dist, a.angle, a.covis_frac, a.covis_angle)
                note = f"gt={gtp.name} dist={dist:.3g}"
                if est_path is not None and pathlib.Path(est_path).exists():
                    z = np.load(est_path, allow_pickle=True)
                    est = {k[4:]: np.asarray(z[k], float) for k in z.files if k.startswith("est_")}
                    note += " +est"
            except Exception as ex:  # noqa: BLE001
                note = f"GT unusable ({ex})"
        else:
            note = "no GT matched"
        recs = analyse(rep, truth, a.row, est, gt, owner)
        for r in recs:
            r.update(report=p.stem, scene=rep.get("scene"), backbone=rep.get("backbone"))
        all_recs += recs
        S = summarise(recs, truth is not None)
        rows = {r["method"]: r for r in rep.get("rows", [])}
        pt = None
        if "chained" in rows and a.row in rows:
            pt = (rows["chained"].get("auc_within"), rows[a.row].get("auc_within"), rows["chained"].get("ate_rmse"), rows[a.row].get("ate_rmse"))
        S.update(report=p.stem, scene=rep.get("scene"), backbone=rep.get("backbone"), note=note,
                 auc_within_raw=(pt[0] if pt else None), auc_within_row=(pt[1] if pt else None),
                 ate_raw=(pt[2] if pt else None), ate_row=(pt[3] if pt else None))
        per.append(S)
        line = (f"{p.stem}: windows {S['windows']}, proposals in {S['windows_with_proposals']}, sites {S['sites_proposed']} "
                f"(verified {S['sites_verified']}; fail rot/dir/extent {S['site_fail_rot']}/{S['site_fail_dir']}/{S['site_fail_extent']}), "
                f"closures tested {S['closures_tested']} accepted {S['accepted']} (rejected rot/pos/scale "
                f"{S['rejected_rot']}/{S['rejected_pos']}/{S['rejected_scale']})")
        if truth is not None:
            line += (f"; GT revisit windows strict {S['windows_with_gt_revisit']} / covis {S['windows_with_covis_revisit']}; "
                     f"closure precision strict {fmt(S['closure_precision'])} 2x {fmt(S['closure_precision_2x'])} covis {fmt(S['closure_precision_covis'])}; "
                     f"recall strict {fmt(S['closure_recall'])} covis {fmt(S['closure_recall_covis'])}; missed no-prop/no-verified/budget/other "
                     f"{S['missed_no_proposal']}/{S['missed_no_verified']}/{S['missed_budget']}/{S['missed_other']}")
        if "closures_scored" in S:
            line += (f"; useful {S['useful']}/{S['closures_scored']} harmful {S['harmful']} (window-pair placement error "
                     f"{S['pos_before_mean']:.3f} -> {S['pos_after_mean']:.3f}; angular {S['ang_before_mean']:.1f} -> {S['ang_after_mean']:.1f} deg)")
        if pt and pt[0] is not None:
            line += f"; AUC_within raw {pt[0]:.1f} -> {pt[1]:.1f}; ATE {pt[2]:.3f} -> {pt[3]:.3f}"
        print(line + f"  [{note}]")
    has_gt = bool(all_recs) and all("gt_revisit" in r for r in all_recs)
    agg = summarise(all_recs, has_gt)
    print("\nPOOLED over", len(paths), "reports:")
    for k, v in agg.items():
        print(f"  {k:<34} {fmt(v)}")
    if any("useful" in r for r in all_recs):
        print("\nACCEPTED CLOSURES by recorded feature (pooled):")
        print(breakdown(all_recs))
    if a.md:
        keys = ["report", "windows", "windows_with_proposals", "sites_proposed", "sites_verified", "closures_tested", "accepted",
                "rejected_rot", "rejected_pos", "rejected_scale", "anchored_passes_attempted"]
        if has_gt:
            keys += ["windows_with_gt_revisit", "windows_with_covis_revisit", "closure_precision", "closure_precision_2x",
                     "closure_precision_covis", "closure_recall", "closure_recall_covis", "false_closures_covis",
                     "missed_no_proposal", "missed_no_verified", "missed_budget", "recall_at_1", "recall_at_5"]
        keys += ["closures_scored", "useful", "harmful", "pos_before_mean", "pos_after_mean", "ang_before_mean", "ang_after_mean",
                 "auc_within_raw", "auc_within_row", "ate_raw", "ate_row"]
        lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for S in per + [dict(agg, report="POOLED")]:
            lines.append("| " + " | ".join(fmt(S.get(k)) for k in keys) + " |")
        pathlib.Path(a.md).write_text("\n".join(lines) + "\n"); print("markdown ->", a.md)
    if a.csv:
        keys = sorted({k for r in all_recs for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_recs)
        print("records ->", a.csv)


if __name__ == "__main__":
    main()
