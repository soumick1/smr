# smr_updates178 — a CALIBRATED row for Table 2 (7-Scenes seq-01)

## Why it is not just "+SMR with known K"
The pose-based rows (chained / +SMR / +SMR+PGO) never use intrinsics: VGGT-Omega's camera head outputs poses, and every
SMR operation is a Sim(3) fit on shared-frame poses. Feeding known K there changes nothing; such a row would be a relabel.
Intrinsics enter only the DENSE-EDGE variant (`smr_pgo_dense`: every junction/closure edge re-fitted from depth maps
unprojected with intrinsics) -- the analogue of MASt3R-SLAM's calibrated mode. So the calibrated row is
**+SMR+PGO, dense edges, known intrinsics (585, 585, 320, 240 @ 640x480, rescaled to the depth resolution)**, reported
next to its uncalibrated twin (dense edges with predicted intrinsics), so the effect of calibration itself is visible.

## Code
pilot_a `--known-K FX FY CX CY [--known-K-res W H]` replaces the predicted intrinsics inside the dense re-measurement only;
prints predicted vs known K once per run (`[calibrated] depth WxH; predicted f=(..) c=(..) -> known f=(..) c=(..)`) so the
intrinsic error being removed is on record. The report records `known_K`.

## Run (GPU 0; passes cached; ~2-4 min per scene per variant -> ~40 min)
```bash
cd ~/smr && unzip -o smr_updates178.zip
screen -dmS cal bash -c "$ENV && CUDA_VISIBLE_DEVICES=0 bash scripts/run_7scenes_calibrated.sh > outputs/reports/7scenes_calibrated.log 2>&1"
# when done: the script prints the per-scene table (raw / +SMR / +PGO / +PGO dense pred-K / +PGO dense known-K); then
tar czf outputs/7scenes_calibrated.tgz outputs/7scenes_calibrated/*.json outputs/reports/7scenes_calibrated.log
```
Send the tgz. Table 2 gets the row "+SMR+PGO (dense edges, known intrinsics)" in the calibrated block and the twin
"(dense edges)" in the uncalibrated block; App. D says which rows use intrinsics.

## Predictions (recorded)
P75: predicted focal within 3 % of the known one (VGGT-Omega estimates 7-Scenes intrinsics well) -> the calibrated dense row
within 0.002 m of the uncalibrated dense row on the mean; P76: both dense rows within +-0.003 m of smr_pgo (0.029): dense
edges neither help nor hurt on 7-Scenes at this scale (the pose fit is already good; dense re-fit adds depth noise).
