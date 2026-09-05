# smr_updates124 — where the memory can act in VGGT's Tables 2–4, and where it cannot

## 1. Scored (scoreboard 5.5/16)

- **P14 half.** ETH3D with predictions kept at masked pixels: plain Umeyama **0.686 / 0.553 / 0.619** vs published 0.873 / 0.482 / 0.677 (masked: 0.419 / 0.601 / 0.510). Comp in range, Overall within 0.06 of the paper. Table 3 is reproduced up to the authors' frame sampling; the only protocol degree of freedom left is whether predictions at masked pixels are kept. We report the masked row (protocol wording) and footnote the unmasked one.
- **P16 miss.** courtyard: chained = smr = smr_pgo byte-identical (pilot_a: 0 loops, no GT revisit pairs among 38 frames). Pose-level memory has nothing to correct on these benchmark sets.
- **DTU 23-scan baseline** (`scripts/dtu_summary.py outputs/reports/dtu_baseline_v123.log`):
  - depth branch, GT cams, geo≥3, C>2, ICP: **1.093 / 0.389 / 0.741** (KIT VGGT-d 1.53 / 0.40 / 0.97; better than KIT on 3 of the 4 shared scans).
  - point branch, C>2, predicted cams, ICP: **1.086 / 1.051 / 1.068** — same Acc, Comp 2.5× worse than the depth route on 22/23 scans and 2× worse than KIT's VGGT-p on the shared scans (24/110/114/118: ours 0.95/0.94/0.83/1.25 vs 0.65/0.45/0.47/0.61). A **coverage** defect of our point-head route (holes from C>2, or view misplacement through predicted cameras), not an accuracy one — diagnosed in block I.

## 2. Analysis: the three tables and the memory

The pass-through invariant (camera pose, N=10/50) now holds for point maps: within one window, backbone+SMR ≡ backbone. VGGT's Table 2 (49 DTU views: one window), Table 3 (10 ETH3D frames: one window) and Table 4 (ScanNet-1500: two frames) are all *inside* the window, and courtyard showed that even the all-view ETH3D sets (14–76 images) contain no revisits for a pose-level memory to exploit. Pose-level SMR/PGO cannot move these tables, and should not: that is the guaranteed floor. Table 4 stays a floor check only (a two-frame problem has no memory; using other test pairs of the same scene would be leakage).

What the logs *do* show, and what the paper's read step ("recognize, re-measure, verify") is built for, are two **content-level** errors present inside these protocols:

| error | evidence | memory operation |
|---|---|---|
| per-pass point-vs-camera **scale bias**: a pass placed by its cameras carries mis-scaled content | forced 16-view windows: ICP scale 1.028 (native 1.010), identical across rows → belongs to the passes; ETH3D failure scenes: Umeyama scale 0.68–1.18 | **re-measure on content**: compare the new pass's point maps on already-stored views with the cards, solve the per-pass depth scale about the pass cameras (closed form, trimmed); `--content-align` |
| **per-read noise** of a single forward pass (ordering / reference-frame dependent) | native Acc 0.68 on scan1 vs 0.39 published with unreported post-processing | **repeated reads**: K orderings of the same views written to the same scaffold, returned as the consensus read; `--reads K` |

Both operate on the full 49-view DTU set and the same 10 ETH3D frames, so they are *within* the published protocols. If they move Acc, Tables 2–3 gain an honest "+SMR read" row; if they do not (P17 kill), the tables carry the floor rows and the memory's contribution stays where the pose results already place it (long sequences, streaming backbones).

CPU dry run (biased fake backbone: 0/+2/+4 % per pass): camera placement alone fuses to 0.060 m plane error; `--content-align` recovers scale ×0.9804 = 1/1.02 exactly (residual 0.0000) with the pass's own cameras or GT cameras; the free-Sim(3) model only approximates a depth-scale error (kept as `--content-model sim3`).

## 3. Changed in v124 (`python tests/test_points_suite_dryrun.py`)

`points_suite.py`: `--reads K`; `--content-align [--content-model scale|sim3] [--content-trim 0.2] [--content-iters 2]` (works after chained placement, pilot placement, and in `--gt-cams` mode — scale only, about GT cameras); `--gt-cams` now also places the point head (each view's points taken to its predicted camera frame, then to the GT camera); prints the valid-pixel fraction per context. `scripts/dtu_summary.py` (new).

## 4. Runs (predictions recorded)

Common helper (evaluates the **fused** cloud; `single.ply` = read 0 = the native pass):
```bash
cd ~/smr && unzip -o smr_updates124.zip && source .venv/bin/activate && python tests/test_points_suite_dryrun.py | tail -1
runF() { gpu=$1; scan=$2; tag=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu python experiments/points_suite.py --gt data/gt/dtu/scan${scan}.npz --backbone vggt \
    --w 49 --overlap 0 --k-ctx 1 --stride 1 "$@" --out-dir outputs/points/dtu_s${scan}_v124_$tag 2>&1 | grep -E "content-align|carry a point|fused:"
  echo "== $tag scan$scan ==" | tee -a outputs/reports/dtu_v124.log
  python scripts/dtu_eval.py --pred outputs/points/dtu_s${scan}_v124_$tag/fused.ply --scan $scan \
    --sampleset ~/data/dtu/SampleSet/MVS\ Data --points-dir ~/data/dtu/Points/stl --icp 8 | tee -a outputs/reports/dtu_v124.log; }
```

### F. In-window memory read, DTU scan1 (GPU 1, ~12 min) — P17
```bash
runF 1 1 ph_reads4        --points-from pointhead --conf-abs 2.0 --reads 4
runF 1 1 ph_reads4_ca     --points-from pointhead --conf-abs 2.0 --reads 4 --content-align
runF 1 1 dp_reads4_ca     --conf-abs 2.0 --geo-views 3 --gt-cams --geo-cams gt --reads 4 --content-align
```
**P17:** `ph_reads4_ca` Acc ≤ **0.60** (native 0.681), Overall ≤ **0.52** (native 0.560); `ph_reads4` without content-align worse than with (scale spread). Kill: |ΔAcc| < 0.03 → the in-window residual is systematic (ordering-independent); no in-window story, tables carry floors only.

### G. Beyond-window with content re-measure, DTU scan1 (GPU 2, ~8 min) — P18
```bash
for row in chained smr smr_pgo; do
  CUDA_VISIBLE_DEVICES=2 python experiments/points_suite.py --gt data/gt/dtu/scan1.npz --backbone vggt \
    --pilot outputs/reports/pilotA_dtu_scan1_vggt_est.npz --row $row --stride 1 --points-from pointhead --conf-abs 2.0 \
    --content-align --out-dir outputs/points/dtu_s1_matrix_${row}_ca 2>&1 | grep -E "content-align|fused:"
  echo "== ${row}_content scan1 ==" | tee -a outputs/reports/dtu_v124.log
  python scripts/dtu_eval.py --pred outputs/points/dtu_s1_matrix_${row}_ca/fused.ply --scan 1 \
    --sampleset ~/data/dtu/SampleSet/MVS\ Data --points-dir ~/data/dtu/Points/stl --icp 8 | tee -a outputs/reports/dtu_v124.log
done
```
**P18:** chained+content Overall 0.867 → ≤ **0.70** (≥ 50 % of the window loss recovered), ICP scale back toward 1.01; the three rows stay within 0.05 of each other (junctions were already exact).

### H. ETH3D failure scenes, in-window reads under the 10-frame protocol (GPU 0, ~10 min) — P19
```bash
for sc in meadow relief terrains courtyard; do
  CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt data/gt/eth3d/${sc}.npz --backbone vggt --n-views 10 --view-seed 0 \
    --w 10 --overlap 0 --k-ctx 1 --stride 1 --reads 4 --content-align --save-maps \
    --out-dir outputs/points/ethrep124_${sc} 2>&1 | grep -E "content-align|fused:"
  echo "== $sc reads4 ==" | tee -a outputs/reports/eth_reads.log
  python scripts/eth3d_pointmap_eval.py --maps outputs/points/ethrep124_${sc}/maps.npz --scene-dir ~/data/eth3d/$sc \
    --gt official --source fused --json outputs/reports/eth_reads.jsonl | grep " Acc " | tee -a outputs/reports/eth_reads.log
done
```
**P19:** plain-Umeyama Overall: meadow 2.34 → ≤ 1.2, relief 1.06 → ≤ 0.8, terrains 0.93 → ≤ 0.6; courtyard unchanged within 0.02. If the failure scenes do not move, their error is a deterministic function of the image set and no in-window read fixes it.

### I. Point-head coverage, scan29 (GPU 1, ~6 min) — P20
```bash
runF 1 29 ph_c2_gtcams   --points-from pointhead --conf-abs 2.0 --gt-cams
runF 1 29 ph_c15         --points-from pointhead --conf-abs 1.5
runF 1 29 ph_nofilter    --points-from pointhead
```
**P20:** if `ph_c2_gtcams` brings Comp 2.37 → ≤ 0.8 the defect is view misplacement through predicted cameras (fix: report the point head with GT-camera placement, same ✓ column as the depth row); if only `ph_c15`/`ph_nofilter` fix it, the defect is threshold holes (fix: C>1.5). Then rerun the 23-scan point-head baseline with the winning recipe.

## 5. Recommendation for the tables
- Table 2 (DTU): rows = published (their protocol) | KIT independent | ours: VGGT-d, VGGT-p (fixed recipe) as floors; "+SMR read" rows if P17/P18 hold; streaming backbones raw vs +SMR (their own memory is write-once; ours corrects it) as the row where a memory belongs.
- Table 3 (ETH3D): 10-frame protocol rows (masked; unmasked footnoted) as floors; "+SMR read" if P19 holds; all-view windowed rows with content re-measure as the beyond-window study.
- Table 4 (ScanNet-1500): floor only, lowest priority; a two-frame problem has no memory.
