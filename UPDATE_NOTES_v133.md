# smr_updates133 — the merged main table: Table-1 rows × {camera pose, DTU, ETH3D}

## 1. What changed
- **Table design** (`scripts/build_main_table.py` → `paper/tab_main.tex`, replaces Table 1, keeps `\label{tab:pose}`):
  rows = Table 1's rows for all 8 backbones; columns = three task groups with a consistent header:
  row 1 task + metric ("Camera pose: AUC@30↑" | "Dense MVS: DTU, Chamfer mm↓" | "Point maps: ETH3D, Chamfer m↓"),
  row 2 input length (N=10…200 | 49 views, 22 scans | 10 frames, 13 scenes), row 3 dataset or sub-metric
  (RE10K CO3D | Acc Comp Overall). No GT-cams column, no published rows (anchors live in the caption).
  Camera cells are taken verbatim from the original Table 1 (`paper/tab_pose_camera.tex`, markers kept).
  Row semantics for the dense columns: frozen backbones raw = native pass, +SMR = 4-ordering read, +SMR+PGO = read
  (no junction graph in one window); streaming: native = one causal pass, (raw, windowed) = 16-view windows fused by
  their median, +SMR = windows + content re-measure, +SMR+PGO = same. Overall is bold where the memory improves the row.
- **Backbones**: every point-map backbone now exposes its native point map (`assemble()` computes `world_points` from
  `pts_local`; π³ from its local points) so `--points-from auto` picks the point head where it exists (VGGT, π³, Fast3R)
  and depth×camera otherwise (DUSt3R, MASt3R, VGGT-Ω, StreamVGGT, STream3R). VERIFY-ON-SERVER: first scan of each.
- **Confidence guard**: `--conf-abs 2.0` falls back to the top 68 % by percentile when the cut would keep < 5 % (π³'s
  sigmoid confidence); printed once. Tests: `tests/test_native_pointmaps.py`.
- **Order-invariant backbones** (DUSt3R, MASt3R: pairwise + global alignment; π³: permutation-equivariant): the
  4-ordering read is the identity → they run one read; +SMR = raw by construction (stated in the caption).
- `scripts/run_dense_matrix.sh GPU bb...`: resume-safe launcher (one GPU, backbones sequential); one 4-read run per
  scan yields both rows (single.ply = native, fused.ply = read); streaming backbones also get the windowed rows.
- `scripts/seed_matrix_from_existing.sh`: VGGT rows from the clouds already on disk, re-scored with the current
  evaluator so all backbones share one gauge (CPU, ~1.5 h).
- `paper/downstream.tex` is now prose only (points to the DTU/ETH3D columns of Table 1); `paper/downstream_appendix.tex`
  keeps the policy ablation. Sandbox Overleaf copy compiles: 0 errors, 0 overfull boxes, 14 pages.

## 2. Launch (all three GPUs; ~7 h wall)
```bash
cd ~/smr && unzip -o smr_updates133.zip && source .venv/bin/activate
python tests/test_native_pointmaps.py | tail -1 && python tests/test_points_suite_dryrun.py | tail -1
screen -dmS seed bash -c 'bash scripts/seed_matrix_from_existing.sh 2>&1 | tee outputs/reports/matrix_seed.log'        # CPU
screen -dmS mx0  bash -c 'bash scripts/run_dense_matrix.sh 0 vggt_omega streamvggt 2>&1 | tee outputs/reports/matrix_gpu0.log'
screen -dmS mx1  bash -c 'bash scripts/run_dense_matrix.sh 1 pi3 dust3r stream3r  2>&1 | tee outputs/reports/matrix_gpu1.log'
screen -dmS mx2  bash -c 'bash scripts/run_dense_matrix.sh 2 mast3r fast3r        2>&1 | tee outputs/reports/matrix_gpu2.log'
```
Progress: `tail -n 3 outputs/reports/matrix_gpu*.log`. Anytime: `python scripts/build_main_table.py --matrix outputs/reports/matrix --out paper/tab_main.tex`
(missing cells print `--`; partial rows carry the scan/scene count as a superscript). Estimated per backbone: 4-read
backbones ~2.5 h (DTU) + 20 min (ETH3D); 1-read ~1–2 h; streaming windows +2 h each.

Check after the first scan of each backbone (`outputs/points/matrix/<bb>_dtu_s1.log`): the `points-from auto -> ...`
line, the confidence guard line if any, and `ctx 0: xx% of pixels carry a point`. STream3R may OOM at 49 views
(Table 1: quadratic mask) — its native cells then read "OOM" and the windowed rows carry it.

## 3. Predictions (recorded)
- **P37** order-dependent backbones (VGGT-Ω, Fast3R, StreamVGGT, STream3R): the read improves DTU Overall on ≥ 15/22
  scans each and the 13-scene ETH3D mean; order-invariant backbones: identical rows (0.000 difference).
- **P38** STream3R native 49-view pass fails (OOM) on ≥ 1 scan; StreamVGGT native runs on all 22.
- **P39** no backbone's raw DTU Overall is below VGGT's 0.441 except possibly π³ (predicted 0.40–0.60); DUSt3R raw > 1.0.
- **P40** streaming windowed rows: re-measure better than raw on ≥ 15/22 DTU scans for both.

## 4. Overleaf (after the matrix)
1. Replace the Table-1 environment (`\begin{table}[t] … \label{tab:pose} … \end{table}`) with `\input{tab_main}` and
   upload `paper/tab_main.tex` (rebuilt from the full matrix).
2. Replace the "Remaining tasks…" line with `\input{downstream}`; append `\input{downstream_appendix}` to appendix.tex;
   upload both files; add the `langendoerfer2026vggt` bib entry.
3. Delete nothing else; `tab:pose` references keep working.

## 5. Next: NVS training on GSO (per the user), once the matrix is launched.
