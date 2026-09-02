# smr_updates122 — VGGT Table 2/3 reproduction: the missing protocol steps

Scope: reproduce VGGT's DTU Table 2 (0.389 / 0.374 / 0.382 mm) and ETH3D Table 3
(0.873 / 0.482 / 0.677 m, "Ours Depth+Cam") closely enough to trust the harness,
then run backbone × {raw, +SMR, +SMR+PGO} on it. Nothing here changes the memory;
it changes the *instrument* so that it matches the papers.

## 1. Protocol facts (read from the papers/code this session, with sources)

| # | Fact | Source |
|---|------|--------|
| F1 | VGGT DTU = "Multi-view Depth Estimation … Following MASt3R"; VGGT/DUSt3R are the only rows without GT cameras *at inference*. | VGGT §4.2 |
| F2 | MASt3R DTU: matches triangulated with GT cameras, then **"remove spurious 3D points via geometric consistency post-processing [PatchmatchNet]"**, official DTU evaluator. | MASt3R §4.5 |
| F3 | PatchmatchNet filter: reproject to ≥ `geo_mask_thres=5` of 10 source views within **1.0 px** and **1 % relative depth**; fused depth = mean of reference + consistent reprojected depths. | PatchmatchNet `eval.py` (fetched) |
| F4 | DUSt3R DTU (the other ✗ row): "align the predictions to the ground-truth coordinate system … by fixing the [camera] parameters as constants in [global alignment]" → predicted depth unprojected through **GT cameras** = our `--gt-cams`. | DUSt3R §4.5 |
| F5 | MASt3R runs DTU at 384×512 (coarse) → 0.652/0.592/0.622 without coarse-to-fine; VGGT runs at 518. | MASt3R App. C |
| F6 | ETH3D: 10 random frames/scene, **Umeyama** alignment, invalid points removed with the **official masks**, Chamfer Acc/Comp/Overall; Depth+Cam beats the point head. | VGGT §4.3 |
| F7 | ETH3D official GT depth: `<scene>_dslr_depth.7z` → `ground_truth_depth/dslr_images/<name>`, raw float32 on the **distorted** image grid, `inf` = no GT (the mask); calibration THIN_PRISM_FISHEYE in `dslr_calibration_jpg`. | eth3d.net/documentation; robustmvd `compute_eth3d_undistorted.py` |
| F8 | VGGT preprocessing: resize width→518, height→round(H·518/W/14)·14 (anisotropic), centre-crop height to 518 if larger. | vggt `load_and_preprocess_images` |

Consequence for the gap: our DTU Acc (1.1 mm) was measured on **unfiltered** depth. Every
published row in Table 2 — including the ✗ rows — passes a multi-view consistency filter
(F2/F3) or an optimisation that fuses views (F4). Our ETH3D Comp (1.3 m) was measured
against the **full laser scan**; the paper's Comp is against the GT visible in the 10 frames
(F6/F7), and its Umeyama runs on per-pixel correspondences, not camera centres.

## 2. What changed (all CPU-tested; `python tests/test_mvs_fusion.py tests/test_eth3d_gt.py tests/test_points_suite_dryrun.py`)

- `src/smr/eval/mvs_fusion.py` (new): `geo_consistency` (F3, vectorised), `select_sources`
  (10 nearest cameras = pair.txt ranking), `umeyama_trimmed`, `chamfer`.
- `src/smr/eval/eth3d_gt.py` (new): GT point maps on the backbone grid from the official
  depth (fisheye forward projection + per-cell median pooling; F7/F8) or a laser-scan
  z-buffer fallback (`--gt scan`, no download, labelled fallback); `grid_map` implements F8.
- `experiments/points_suite.py`: `--geo-views N --geo-pix 1.0 --geo-rel 0.01 --geo-src 10
  --geo-cams pred|gt --no-geo-avg`, `--save-maps` (per-view point maps for the
  correspondence protocol), GT-K scaling honours the crop (`K_at_grid`), `build_parser()`.
- `src/smr/backbones/vggt.py`: exposes `world_points` (point head) scale-consistent with
  depth/poses → "Ours (Point)" row reproducible.
- `scripts/eth3d_pointmap_eval.py` (new): F6 offline from `maps.npz`; prints pre-align
  (camera Sim(3)), **umeyama** (paper wording), umeyama-trim (diagnostic) + F1@tol.
- `scripts/eth3d_fetch_gt.sh` (new), `scripts/eth3d_pm_summary.py` (new).

## 3. Run — DTU scan1 sweep (GPU 1, ~15 min total)

```bash
cd ~/smr && unzip -o smr_updates122.zip && source .venv/bin/activate
python tests/test_mvs_fusion.py && python tests/test_eth3d_gt.py && python tests/test_points_suite_dryrun.py | tail -1
mkdir -p outputs/reports
run1() { tag=$1; shift
  CUDA_VISIBLE_DEVICES=1 python experiments/points_suite.py --gt data/gt/dtu/scan1.npz --backbone vggt \
    --w 49 --overlap 0 --k-ctx 1 --stride 1 "$@" --out-dir outputs/points/dtu_s1_v122_$tag 2>&1 | grep -E "geo-consistency|single:"
  echo "== $tag ==" | tee -a outputs/reports/dtu_s1_v122.log
  python scripts/dtu_eval.py --pred outputs/points/dtu_s1_v122_$tag/single.ply --scan 1 \
    --sampleset ~/data/dtu/SampleSet/MVS\ Data --points-dir ~/data/dtu/Points/stl --icp 8 | tee -a outputs/reports/dtu_s1_v122.log; }
run1 c40_geo3         --conf-pct 40 --geo-views 3
run1 c40_geo5         --conf-pct 40 --geo-views 5
run1 geo5             --geo-views 5
run1 c40_geo3_pix05   --conf-pct 40 --geo-views 3 --geo-pix 0.5
run1 c40_geo3_noavg   --conf-pct 40 --geo-views 3 --no-geo-avg
run1 gtcams_c40_geo3  --conf-pct 40 --geo-views 3 --gt-cams --geo-cams gt
python scripts/dtu_eval.py --pred outputs/points/dtu_s1_v122_gtcams_c40_geo3/single.ply --scan 1 \
  --sampleset ~/data/dtu/SampleSet/MVS\ Data --points-dir ~/data/dtu/Points/stl | tee -a outputs/reports/dtu_s1_v122.log  # no ICP (DUSt3R-style)
grep -E "^==|^scan1" outputs/reports/dtu_s1_v122.log
```
Reference points: c40 unfiltered + icp8 = 1.098 / 0.416 / 0.757 (v121); published 0.389 / 0.374 / 0.382.

## 4. Run — ETH3D (GPU 0; fallback GT tonight, official GT after the download)

```bash
# A. 13 scenes, their setting (10 frames, seed 0, native pass, no conf filter), maps saved; scored with the scan fallback
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt $f --backbone vggt --n-views 10 --view-seed 0 \
    --w 10 --overlap 0 --k-ctx 1 --stride 1 --save-maps --out-dir outputs/points/ethrep122_${sc} 2>&1 | tail -1
  echo "== $sc ==" | tee -a outputs/reports/eth_pm_scan.log
  python scripts/eth3d_pointmap_eval.py --maps outputs/points/ethrep122_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc \
    --gt scan --json outputs/reports/eth_pm_scan.jsonl | grep " Acc " | tee -a outputs/reports/eth_pm_scan.log
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_pm_scan.jsonl
# B. official GT (F7): ~10 GB download, then re-score without GPU (also --source pointhead for the "Ours (Point)" row)
bash scripts/eth3d_fetch_gt.sh ~/data/eth3d
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  for src in single pointhead; do
    python scripts/eth3d_pointmap_eval.py --maps outputs/points/ethrep122_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc \
      --gt official --source $src --json outputs/reports/eth_pm_official.jsonl | grep " Acc "
  done
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_pm_official.jsonl
```

## 5. Predictions (recorded before the runs; scoreboard 3.0/10)

- P11 DTU scan1 `c40_geo3`: Acc 1.098 → **0.55–0.75**, Comp 0.42 → 0.45–0.60, Overall **0.50–0.65**.
  `geo5` stricter: Acc another −0.05…−0.15, Comp +0.1…+0.2. `noavg` worse than `c40_geo3` by
  0.05–0.15 Acc (averaging is half the reference filter). `gtcams` within ±0.10 of `c40_geo3`.
  Kill criterion: if the best Acc stays > 0.9 the filter is not the gap and the residual is
  518-px depth quality (F5 says a 512-px method with triangulation reaches 0.65 Acc).
- P12 ETH3D seed-0 mean with the **scan** fallback + Umeyama: Comp 1.294 → **0.45–0.70**,
  Acc 0.684 → 0.70–0.95 (GT restricted to the 10 frames' visible surface), Overall **0.60–0.80**
  vs published 0.677. Official GT (B) within ±0.1 of the fallback; the pointhead row worse
  than Depth+Cam (paper: 0.709 vs 0.677).

## 6. Decision tree after the sweep

1. DTU best Overall ≤ 0.6 → adopt that recipe as the **baseline row** ("VGGT, our harness,
   MASt3R post-processing"), run the 23 scans on GPU 1 (~1 h at stride 1), quote published rows
   under *their* protocol, and launch the matrix (§7).
2. DTU best Overall 0.6–0.9 → same, with the gap attributed in the caption to F5 (resolution);
   ask Mengmi whether "same order under a matched post-processing" is the bar.
3. ETH3D Overall within ±0.15 of 0.677 → protocol validated; matrix.

## 7. Matrix (after the baseline; same harness, camera-pose style)

Per backbone in {vggt, vggt_omega, pi3, dust3r, mast3r, fast3r} + {streamvggt, stream3r}:
- DTU: `raw` = 16-view windows chained Sim(3) (beyond-window regime; native 49-view pass is the
  floor row), `+SMR` = `--k-ctx 0 --w 16 --overlap 8` consensus fusion, `+PGO` = final solve;
  all with `--geo-views 3` (same post-processing for every row) and `--save-maps`.
- ETH3D: all views windowed (`--w 16 --overlap 8 --k-ctx 0 --save-maps`), scored with
  `--source fused` under the official GT; the 10-frame VGGT rows stay as the protocol anchor.
Never run the same backbone on the same sequences concurrently (pass cache); one scan set per GPU.
