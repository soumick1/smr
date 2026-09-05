# SMR HANDOVER MANIFEST — VGGT Table 2/3 reproduction chase (2026-09-05, v141)
Read this fully before acting. Rules that are LAW in this project:
code > Overleaf > memory; every patch VERIFIES its landing (assert/grep) before packaging;
predictions recorded BEFORE runs and scored honestly (current scoreboard 3.0/10);
no silent protocol guesses — read paper/code, state assumptions in captions;
incremental logs + resume for any multi-hour queue; never run same backbone+same
sequences concurrently (pass-cache); 3 GPUs available (CUDA 0/1/2).

## MISSION (user directive)
Get VERY CLOSE to VGGT's published DTU Table-2 (0.389/0.374/0.382 mm) and ETH3D
Table-3 (0.873/0.482/0.677 m, "Ours Depth+Cam") BEFORE launching the
backbone x {raw, +SMR, +SMR+PGO} matrix on these benchmarks. Paper: arXiv 2503.11651
(HTML fetchable at arxiv.org/html/2503.11651v1; /mnt/project/VGGT.pdf is CORRUPT).
If needed read the ENTIRE paper incl. supplementary for eval details.

## PROTOCOL FACTS ALREADY EXTRACTED (verbatim-level, do not re-derive)
- ETH3D §4.3: per scene sample 10 random frames; align predicted cloud to GT via
  Umeyama; filter invalid points with OFFICIAL MASKS; Chamfer Acc/Comp/Overall.
  "Depth+Cam" row = unproject predicted depth with predicted cameras (beats point head).
- DTU §4.2: NO GT cameras (x block), "following MASt3R" (-> DUSt3R lineage).
- facebookresearch/vggt repo has NO eval scripts (checked) — training+model only.

## CURRENT BEST REPRODUCTION STATE
- DTU scan1 best recipe: backbone=vggt, native w=49 k-ctx=1, stride 2, --conf-pct 40,
  eval --icp 8  =>  Acc 1.098 Comp 0.416 Overall 0.757 (kept 80.1%). c50: 0.960. 
- DTU 23-scan mean @c50+icp8: Acc 1.348 Comp 1.174 Overall 1.261 (worst: scan29 2.80,
  scan77 2.04, scan62 1.79; w/o scan29 -> 1.19). Gap to 0.382: ~3.3x. TODO: rerun full
  set at c40; try per-scan conf; investigate scan29 (Comp 3.90 = coverage/align fail).
- ETH3D 10-frame repro means (seed/Acc/Comp/Overall): s0 0.684/1.294/0.989,
  s1 ~/1.6?/0.851, s2 ~/?/0.865. Our Acc BEATS published 0.873; Comp ~2.7x worse.
- ETH3D windowed+SMR (all views, our instrument): 0.585/0.346/0.466 — better than
  published 0.677 but different protocol (all views). In draft? NOT yet as table.

## STATE (UPDATE_NOTES_v133.md) -- MERGED MAIN TABLE; DENSE MATRIX LAUNCHING
- Table 1 becomes the merged table (scripts/build_main_table.py -> paper/tab_main.tex, label tab:pose):
  Table-1 rows x {camera AUC@30 (8 cols, verbatim), DTU Acc/Comp/Overall, ETH3D Acc/Comp/Overall}.
  Dense rows: raw = native pass; +SMR = 4-ordering read; +PGO = read. Streaming: native / windowed raw /
  windowed + re-measure. Compiles in the Overleaf copy (0 errors, 0 overfull, 14 pages).
- Dense matrix over 8 backbones: scripts/run_dense_matrix.sh (3 GPUs; markers DONE/OOM/FAILED, result-aware
  resume, OOM recorded and shown as OOM in the table; tests/test_launcher.sh) + seed script for VGGT.
- dtu_eval.py rewritten: multi-pred per call, hashed thinning, binary PLY (suite writes binary now). Same numbers.
- v133 Overleaf zip had STAND-IN StreamVGGT windowed cells -> discarded; v134 zip is real-only.
- MATRIX RUN DONE (Sep 5): all backbones complete; DUSt3R DTU = OOM on 22/22 (complete pair graph). v135 adds
  --backbone-kw + a dust3r memory ladder (swin-5 -> swin-3); rerun block in UPDATE_NOTES_v135.md §3.
  Results received (v134 gauge): DTU column INVALID for non-VGGT backbones (gauge caps refused 6-25% scale
  corrections; 8-18 rejections/row) and a one-sided fit slid vggt_smr scan13. v136 evaluator: symmetric fit,
  robust similarity init, sanity caps only. CPU re-score = scripts/rescore_matrix_dtu.sh (~3-4 h).
  ETH3D valid: VGGT 0.490->0.437, pi3 0.122, DUSt3R 0.742, MASt3R 0.653; read HURTS Fast3R/StreamVGGT (non-
  commensurable orderings). VGGT-Omega 0.047 = contamination caveat (authors' README) -> marked with §.
- v136 re-score: rejections 0-3/row; MASt3R 0.80, VGGT-Omega 0.78->0.68, pi3 1.04, STream3R 1.75->1.22, StreamVGGT
  2.92->2.30, DUSt3R(swin-5) 2.11, Fast3R 4.7. Defect: median fit let a 23%-shrunk cloud win (VGGT scan13 0.91).
  v137: fit = truncated Chamfer mean. v138: FINAL re-score done; VGGT 0.441/0.396 reproduced exactly; final
  Table 1 in iclr2027_overleaf_v138.zip. DENSE BENCHMARK SECTION CLOSED. Next: NVS training on GSO.
  Order-invariant backbones (dust3r, mast3r, pi3) run one read (+SMR == raw by construction).
- Native point maps exposed for all point-map backbones (assemble world_points; pi3); conf guard for sigmoid conf.
- Known VGGT cells: DTU 0.552/0.330/0.441 -> 0.476/0.315/0.396; ETH3D 0.415/0.565/0.490 -> 0.274/0.599/0.437.
- Scoreboard 15.0/38. Open: P37-P40 (matrix). Then: NVS training on GSO.

## INFRASTRUCTURE (persistent sandbox /home/claude; server ssh soumick@zlab-ws3)
- Sandbox repo /home/claude/smr mirrors server ~/smr. Latest shipped:
  smr_updates141.zip (v141). Key tools: experiments/points_suite.py (v120:
  windows+overlap, chained junctions, tau-rel gate, keep-singles, --gt-cams,
  --n-views/--view-seed, --conf-pct), scripts/dtu_eval.py (v121: ObsMask/BB/plane
  protocol, --points-dir, --icp Sim3-ICP; CALIBRATED: reproduces furu/tola/camp
  scan1 orderings exactly), scripts/eth3d_eval.py (v118: mlp parse, voxel-thin GT,
  mean-dist + F1), scripts/dtu_prep.py (v116 index-pairing), dtu_prep_sampleset.py
  (v108), eth3d_prep.py.
- Data on server: ~/data/dtu/{dtu(MVSNet 22 test scans), Points/stl(GT),
  SampleSet/MVS Data(ObsMask, Calibration, reference Points furu/tola/camp)},
  ~/data/eth3d/<13 scenes> incl. dslr_scan_eval/scan_alignment.mlp.
- Draft: the user's Overleaf (uploaded 2026-09-02) has NO DTU/ETH3D content; it ends at
  the SLAM table ("Remaining tasks ... follow"). Both tables must be inserted into THAT
  version (values to carry: native Ω floor 4.298; windowed 3.299; +SMR 2.880 — story:
  windows+memory BEAT native). Sandbox copy: /home/claude/iclr_paper/iclr2027.
- Transcripts: /mnt/transcripts (see journal.txt) for verbatim history.

## IMMEDIATE NEXT ACTIONS (UPDATE_NOTES_v139.md)
1. NVS dry run (§4): 20 objaverse + 5 GSO, render, LOOK at one GSO input view (up-axis check), report s/view.
2. Full fetch + render (§5) on 3 GPUs (~13 h predicted).
3. v140: Gaussian head + trainer + GSO evaluator; then rows raw / +SMR read; Overleaf Table 7 rows.
