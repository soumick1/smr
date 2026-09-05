# smr_updates134 — launcher hardened (OOM-aware, resume-exact), evaluator 2-3x faster, table builder OOM-aware

## 0. Please discard the v133 Overleaf zip
Its `tab_main.tex` carried STAND-IN StreamVGGT windowed cells (VGGT's windowed numbers relabelled to exercise
the builder's streaming rows). The v134 Overleaf zip is rebuilt from real data only: VGGT rows filled, every
other dense cell `--` until the matrix runs.

## 1. Launcher `scripts/run_dense_matrix.sh` (tested end-to-end with stubs: `bash tests/test_launcher.sh`)
- Every unit (scan or scene) leaves a marker next to its output: `DONE`, `OOM`, or `FAILED`. Finished units are
  skipped; OOM/FAILED units are **not retried** on resume (`RETRY_FAILED=1` retries them). OOM is detected from
  the suite log ("out of memory", `OutOfMemoryError`, cuBLAS alloc failures) and recorded in the result logs as
  `scanN: OOM` (DTU) or `{"status": "OOM"}` lines (ETH3D) so the table shows **OOM**, not `--`.
- Resume is result-aware: a unit counts as evaluated only if a `scanN: Acc` line (or a `umeyama` json line)
  exists — a crashed evaluation or an earlier OOM mark is re-evaluated when the cloud is there.
- Both clouds of a scan (native + read; or windowed raw + re-measure) are scored in ONE evaluator call.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation OOMs; timestamps on every line.

## 2. Evaluator `scripts/dtu_eval.py` (rewritten; same protocol and numbers — equivalence test vs v129 passes)
- Accepts several `--pred` with `--tags`: the scan, masks, plane, normals and KD-trees are loaded once
  (previously twice per scan).
- Hashed voxel thinning: 12.6 s → 2.1 s on 7 M points. Binary little-endian PLY read/write (the suite now writes
  binary; both readers handle ASCII and binary, with or without `plyfile`): 7 M points read in 0.5 s (ASCII: ~9 s
  to write, ~2 s to read). Expected saving: ~3 min per scan → 4-read DTU backbone ≈ 1.5–2 h instead of 3.
- Gauge unchanged (best of camera / plain / coarse-initialised refinement by fit quality).

## 3. Builder `scripts/build_main_table.py`
- `OOM` cells when no unit of a row finished and at least one ran out of memory; superscript count when a row is
  incomplete; default camera source `paper/tab_pose_camera.tex` (the original Table 1, saved).

## 4. Launch (unchanged commands; ~7 / 6 / 8 h with the faster evaluator)
```bash
cd ~/smr && unzip -o smr_updates134.zip && source .venv/bin/activate
python tests/test_dtu_eval_icp.py | tail -1 && python tests/test_native_pointmaps.py | tail -1 && bash tests/test_launcher.sh | tail -1
screen -dmS seed bash -c 'bash scripts/seed_matrix_from_existing.sh 2>&1 | tee outputs/reports/matrix_seed.log'
screen -dmS mx0  bash -c 'bash scripts/run_dense_matrix.sh 0 vggt_omega streamvggt 2>&1 | tee outputs/reports/matrix_gpu0.log'
screen -dmS mx1  bash -c 'bash scripts/run_dense_matrix.sh 1 pi3 stream3r          2>&1 | tee outputs/reports/matrix_gpu1.log'
screen -dmS mx2  bash -c 'bash scripts/run_dense_matrix.sh 2 mast3r fast3r dust3r  2>&1 | tee outputs/reports/matrix_gpu2.log'
```
Progress: `tail -n 2 outputs/reports/matrix_gpu*.log`; failures: `ls outputs/points/matrix/*/OOM outputs/points/matrix/*/FAILED 2>/dev/null`.
Table any time: `python scripts/build_main_table.py --matrix outputs/reports/matrix --out paper/tab_main.tex`.
After the first scan of each backbone check `outputs/points/matrix/<bb>_dtu_s1.log` for `points-from auto ->` and any `[conf]` guard line.

## 5. Predictions unchanged (P37–P40, notes v133 §3).
