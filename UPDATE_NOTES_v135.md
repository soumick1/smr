# smr_updates135 — matrix complete except DUSt3R/DTU (OOM); the fix and the rerun

## 1. Run outcome (progress logs)
All 8 backbones finished DTU + ETH3D; streaming windowed rows finished (StreamVGGT 02:40, STream3R 01:52).
**DUSt3R on DTU: OOM on all 22 scans** (complete pair graph over 49 views = 1,176 pairs; the global aligner keeps every
pairwise point map on the GPU). Its ETH3D (10-frame) row is fine. The launcher marked each scan `OOM`, did not retry,
and the table shows `OOM` for that row — as designed. Everything else: no OOM/FAILED markers reported.

## 2. Fix (v135)
- `points_suite.py --backbone-kw key=value` (repeatable) passes constructor kwargs to a backbone
  (e.g. `scene_graph=swin-5`; DUSt3R's documented knob for larger image sets).
- Launcher: **memory ladder** for DUSt3R on DTU — complete graph → `swin-5` → `swin-3` on OOM; the rung used is
  recorded in `<outdir>/GRAPH` and announced in the progress line. `LADDER_SKIP_COMPLETE=1` starts at `swin-5`
  for a rerun whose complete-graph OOM is already known (saves ~1 h of re-failing). Non-OOM failures are not retried.
  Covered by `tests/test_launcher.sh` (stub OOMs on the complete graph, succeeds on swin-5).
- Caption to add for the table: "DUSt3R on DTU uses a sliding-window pair graph (window 5); the complete graph over
  49 views exceeds 46 GB." ETH3D (10 frames) keeps the complete graph.

## 3. Rerun DUSt3R/DTU (GPU 2, ~2.5 h; ETH3D units are DONE and skipped)
```bash
cd ~/smr && unzip -o smr_updates135.zip && source .venv/bin/activate && bash tests/test_launcher.sh | tail -1
screen -dmS mx2 bash -c 'RETRY_FAILED=1 LADDER_SKIP_COMPLETE=1 bash scripts/run_dense_matrix.sh 2 dust3r 2>&1 | tee -a outputs/reports/matrix_gpu2.log'
```
If swin-5 also OOMs the ladder drops to swin-3 automatically; if that fails too, the row stays `OOM` and the caption says so.

## 4. Please send the RESULT files (the progress logs do not contain numbers)
```bash
python scripts/build_main_table.py --matrix outputs/reports/matrix --out paper/tab_main.tex
tar czf outputs/reports/matrix_results.tgz outputs/reports/matrix/*.log outputs/reports/matrix/*.jsonl paper/tab_main.tex
```
Attach `matrix_results.tgz` (a few MB). I will score P37–P40, write the caption against the real numbers, and return the
Overleaf zip with the completed Table 1; the DUSt3R DTU cells follow after the rerun.

## 5. Efficiency note for the next round (not applied to running jobs)
Windowed ETH3D evaluations took 1–3 h on the large scenes because the GT point-map cache is per output directory
(raw and re-measured variants each rebuild it) and the Chamfer runs on 10–14 M points; a per-scene cache is the fix.
