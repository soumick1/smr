# smr_updates123 — the reproduction chase is closed; the harness is the instrument

## 1. What the v122 runs said (scored honestly; scoreboard 3.5/12)

**DTU scan1 (P11: MISS).** Predicted Acc 0.55–0.75 after the PatchmatchNet filter; measured:

| recipe (all stride 1, +ICP-8) | kept | Acc | Comp | Overall |
|---|---|---|---|---|
| c40 unfiltered (v121) | — | 1.098 | 0.416 | 0.757 |
| c40 + geo≥3 | 68.6 % | 1.065 | 0.472 | 0.768 |
| c40 + geo≥5 | 61.5 % | 1.075 | 0.534 | 0.804 |
| geo≥5 (no conf) | 61.5 % | 1.004 | 0.428 | 0.716 |
| c40 + geo≥3, 0.5 px | 62.2 % | 1.105 | 0.519 | 0.812 |
| c40 + geo≥3, no averaging | 68.6 % | 1.160 | 0.433 | 0.797 |
| **GT cams + c40 + geo≥3 (gt)** | 68.7 % | **0.965** | **0.308** | **0.636** |
| same, no ICP | | 1.455 | 0.570 | 1.012 |

The filter's 1 % relative-depth gate is 7 mm at DTU depth: it removes edge outliers (median support 8/10 views) and is blind to the ~1 mm residual. Averaging is worth 0.1 mm; GT cameras 0.1 mm Acc / 0.16 mm Comp; ICP fixes a 0.5 % global scale error of VGGT's camera baseline (no-ICP 1.455). Comp 0.308 already beats the published 0.374 — **the whole gap is Acc, i.e. the depth itself.** Kill criterion fired.

**ETH3D (P12: half hit).** Official GT and the laser-scan fallback agree to < 0.01 m everywhere (fallback validated). 13-scene means, "Depth+Cam", plain Umeyama: **0.419 / 0.601 / 0.510** (pre-align 0.416/0.428/0.422; trimmed 0.252/0.185/0.218); point head 0.415/0.572/0.494. Published: 0.873 / 0.482 / 0.677. Comp in range, Acc 2× *better* than published. The mean is carried by meadow/relief/terrains, where the per-pixel Umeyama scale is 0.68–1.18 — VGGT's depth scale and camera-baseline scale disagree by 15–30 % on those 10-frame sets; that is the failure mode, not the harness.

## 2. Independent reproduction (Langendörfer, Landgraf, Ulrich; KIT; arXiv 2606.16479, June 2026)

Feed-forward VGGT, official DTU protocol (evaluation masks, 0.2 mm downsampling, **bounding-box Sim(3) + ICP refinement minimising Accuracy** — i.e. our `--icp`), 15 standard scenes, absolute confidence threshold 2.0:

| | Acc | Comp | CD |
|---|---|---|---|
| VGGT-d (depth branch) | 1.53 | 0.40 | 0.97 |
| VGGT-p (point-map branch) | 0.74 | 0.44 | 0.59 |
| published VGGT (Table 2) | 0.389 | 0.374 | 0.382 |

Quote: *"Wang et al. report 0.38 mm … we suspect they perform bundle adjustment in post-processing … the experiments cannot be reproduced exactly."* Our scan1 depth-branch numbers (0.97–1.07 Acc, 0.31–0.47 Comp, 0.64–0.77 CD) sit in their regime. Two of their findings we had not used: the **point-map branch is 2× more accurate than the depth branch on DTU** (opposite to ETH3D), and the meaningful confidence cut is **absolute C > 2.0** (positive loss weight under VGGT's `expp1` parametrisation), not a percentile.

**Decision.** The 0.382 is not reproducible feed-forward by anyone with the published information. The harness is validated against the only independent reproduction; the paper quotes the published rows *under their protocol* and reports every backbone, +SMR and +PGO under **one** harness (official evaluator, ICP Sim(3), C > 2.0), with the KIT numbers as the external anchor in the caption. Reproduction work stops here; the matrix starts.

## 3. What changed in v123 (CPU-tested: `python tests/test_points_suite_dryrun.py`)

- `points_suite.py`: `--points-from pointhead` (VGGT-p route; the point head is exported scale-consistently by the v122 wrapper), `--conf-abs 2.0` (absolute confidence, any branch), and **`--pilot <npz> --row <chained|smr|smr_pgo>`**: contexts = the pose pipeline's chunks, each pass placed by a Sim(3) fit onto the row's trajectory, GT frame via an ATE-style Sim(3) over all keyframes. Depth is identical across rows by construction — the rows differ only in how the passes are placed, exactly as in the pose table.
- `pilot_a.py`: `--save-est <npz>` writes per-row trajectories + keyframes + chunks + paths (verified in simulated mode).
- Dry run covers: point head + C>2 (exact plane, 96 % kept), pilot rows (drifted `chained` 0.081 m vs exact `smr` 0.000 m, consensus rejecting the disagreeing overlaps).

## 4. Runs

### A. DTU scan1, point-map branch (GPU 1, ~8 min) — P13
```bash
cd ~/smr && unzip -o smr_updates123.zip && source .venv/bin/activate && python tests/test_points_suite_dryrun.py | tail -1
run1() { tag=$1; shift
  CUDA_VISIBLE_DEVICES=1 python experiments/points_suite.py --gt data/gt/dtu/scan1.npz --backbone vggt \
    --w 49 --overlap 0 --k-ctx 1 --stride 1 "$@" --out-dir outputs/points/dtu_s1_v123_$tag 2>&1 | grep -E "geo-consistency|single:"
  echo "== $tag ==" | tee -a outputs/reports/dtu_s1_v123.log
  python scripts/dtu_eval.py --pred outputs/points/dtu_s1_v123_$tag/single.ply --scan 1 \
    --sampleset ~/data/dtu/SampleSet/MVS\ Data --points-dir ~/data/dtu/Points/stl --icp 8 | tee -a outputs/reports/dtu_s1_v123.log; }
run1 ph_c2        --points-from pointhead --conf-abs 2.0
run1 ph_c2_geo3   --points-from pointhead --conf-abs 2.0 --geo-views 3
run1 d_c2_geo3    --conf-abs 2.0 --geo-views 3 --gt-cams --geo-cams gt
grep -E "^==|^scan1" outputs/reports/dtu_s1_v123.log
```
**P13:** `ph_c2` Acc **0.55–0.80**, Comp 0.35–0.50, Overall **0.45–0.65** (KIT 15-scene mean 0.74/0.44/0.59; scan1 is easier than the mean). If the point head beats the depth route by ≥ 0.2 mm Acc it becomes the DTU baseline branch (KIT's finding); `d_c2_geo3` within ±0.05 of v122's `gtcams_c40_geo3` (0.636).

### B. ETH3D: does the published Acc come from unmasked predictions? (no GPU, ~5 min) — P14
```bash
for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
  python scripts/eth3d_pointmap_eval.py --maps outputs/points/ethrep122_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc \
    --gt official --pred-all --json outputs/reports/eth_pm_official_predall.jsonl | grep "^umeyama "
done
python scripts/eth3d_pm_summary.py outputs/reports/eth_pm_official_predall.jsonl
```
**P14:** plain-Umeyama mean with all predictions kept: Acc **0.75–1.10** (published 0.873), Comp 0.55–0.62, Overall **0.65–0.85** (published 0.677). If it lands there, Table 3 is reproduced up to their unknown frame sampling and the ETH3D protocol question is closed; if Acc stays < 0.5 the remaining difference is their frame sampling, which we cannot recover, and we report ours.

### C. Full DTU baseline, both branches (GPU 1, ~2 h; start after A picks the recipe)
```bash
PH="--points-from pointhead --conf-abs 2.0"                    # VGGT-p
DP="--conf-abs 2.0 --geo-views 3 --gt-cams --geo-cams gt"        # VGGT-d (adjust to A's winner)
for f in data/gt/dtu/scan*.npz; do s=$(basename $f .npz | tr -d scan)
  for tag in ph dp; do [ $tag = ph ] && EXTRA="$PH" || EXTRA="$DP"
    CUDA_VISIBLE_DEVICES=1 python experiments/points_suite.py --gt $f --backbone vggt --w 49 --overlap 0 --k-ctx 1 \
      --stride 1 $EXTRA --out-dir outputs/points/dtu_base_${tag}_s${s} 2>&1 | tail -1
    echo "== $tag scan$s ==" >> outputs/reports/dtu_baseline_v123.log
    python scripts/dtu_eval.py --pred outputs/points/dtu_base_${tag}_s${s}/single.ply --scan $s \
      --sampleset ~/data/dtu/SampleSet/MVS\ Data --points-dir ~/data/dtu/Points/stl --icp 8 >> outputs/reports/dtu_baseline_v123.log
  done
done
```

### D. First raw / +SMR / +PGO point-map triple — DTU scan1 (GPU 2, ~15 min) — P15
```bash
CUDA_VISIBLE_DEVICES=2 python experiments/pilot_a.py --gt data/gt/dtu/scan1.npz --backbone vggt --keyframe-stride 1 \
  --chunk 16 --overlap 8 --sites 2 --rows chained,smr,smr_pgo \
  --save-est outputs/reports/pilotA_dtu_scan1_vggt_est.npz --json outputs/reports/pilotA_dtu_scan1_vggt.json 2>&1 | tail -8
for row in chained smr smr_pgo; do
  CUDA_VISIBLE_DEVICES=2 python experiments/points_suite.py --gt data/gt/dtu/scan1.npz --backbone vggt \
    --pilot outputs/reports/pilotA_dtu_scan1_vggt_est.npz --row $row --stride 1 --points-from pointhead --conf-abs 2.0 \
    --out-dir outputs/points/dtu_s1_matrix_$row 2>&1 | grep -E "placement|fused:"
  echo "== matrix $row ==" | tee -a outputs/reports/dtu_s1_matrix.log
  python scripts/dtu_eval.py --pred outputs/points/dtu_s1_matrix_$row/fused.ply --scan 1 \
    --sampleset ~/data/dtu/SampleSet/MVS\ Data --points-dir ~/data/dtu/Points/stl --icp 8 | tee -a outputs/reports/dtu_s1_matrix.log
done
```
**P15:** with 16-view chunks over the 49-view hemisphere, `chained` Overall is worse than the native pass (A's `ph_c2`) by 0.2–1.0 mm; `smr` recovers ≥ 50 % of that gap; `smr_pgo` = `smr` ± 0.05. Kill criterion: if chained and smr agree within 0.05, DTU's 49 regular views do not create the beyond-window regime and the point-map matrix should lean on ETH3D (14–76 views, real revisits) — same recipe with `--save-maps` and `eth3d_pointmap_eval.py --source fused --gt official`.

### E. Then the matrix (3 GPUs, one dataset per GPU, never the same backbone+sequence on two GPUs)
Backbones: `vggt vggt_omega pi3 dust3r mast3r fast3r` + `streamvggt stream3r`; rows `chained smr smr_pgo`; block D per (backbone, scene) with `--backbone` swapped, `--points-from depth` for backbones without a point head. Row = fused.ply (all views through the row's junctions); the native single pass stays as the floor row where the scene fits one window.

## 5. Paper text to carry (for the Overleaf insertion)
DTU caption: published rows quoted under their protocol; "Wang et al.'s 0.382 is not reproduced by feed-forward inference in an independent study (Langendörfer et al., 2026: 0.59 CD point head / 0.97 depth head) nor by us; all rows below share one harness (official evaluator, ICP-refined Sim(3), confidence > 2)." ETH3D caption: "official rendered depth as ground truth and mask; Umeyama on per-pixel correspondences; predictions at masked pixels dropped (kept: see appendix)."
