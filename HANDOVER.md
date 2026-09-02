# SMR HANDOVER MANIFEST — VGGT Table 2/3 reproduction chase (2026-09-03, v125)
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

## STATE (UPDATE_NOTES_v123-125.md)
- Harness validated (KIT independent study). DTU 23-scan VGGT-d 1.093/0.389/0.741; VGGT-p with
  C>2 has a coverage defect = the confidence filter itself (scan29: no filter 0.881 vs C>2 1.953).
- IN-WINDOW MEMORY READ WORKS: scan1 point head, 4 reads + consensus 0.458 vs single 0.560 (P17 hit).
  Content re-measure recovers 92% of the forced-window loss (0.867 -> 0.585, P18 hit). ETH3D relief
  1.06 -> 0.26; meadow deterministic failure; courtyard lost Comp (consensus drop-outs, fixed).
- v125 fixes: content-align symmetric reference (no privileged read; a shadowed variable had
  disabled it), consensus fallback to the earliest witness (a read is never emptier than a pass).
- Pose-level SMR/PGO inert on these tables (pass-through), stated as the floor.
- Scoreboard 9.0/20. Open: P21 (scan1 recipe), P22 (Table-2 rows, 23 scans), P23 (Table-3 rows),
  P24 (ETH3D beyond-window study). Two-view ScanNet-1500: DROPPED by decision.

## INFRASTRUCTURE (persistent sandbox /home/claude; server ssh soumick@zlab-ws3)
- Sandbox repo /home/claude/smr mirrors server ~/smr. Latest shipped:
  smr_updates125.zip (v125). Key tools: experiments/points_suite.py (v120:
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

## IMMEDIATE NEXT ACTIONS (blocks in UPDATE_NOTES_v125.md §3)
1. J (GPU 1, 15 min) -> fix the point-head recipe; then K (GPU 1, 2.5 h) = Table 2 rows.
2. L (GPU 0, 40 min) = Table 3 rows; M (GPU 2, 1 h) = beyond-window ETH3D study.
3. Streaming backbones raw vs +SMR (their write-once memory) on DTU/ETH3D.
4. Draft: insert tab:dtu + tab:eth3d into the uploaded Overleaf (published | KIT | ours | +SMR read).
