# smr_updates147 — smoke test passed; NVS rows fixed to 4 backbones x {raw, +SMR}; launch training

## Smoke test (VGGT, 20 steps, cropped renders)
End to end: backbone -> placement -> head -> gsplat -> LPIPS -> validation. 0.4 s/step (30K steps ≈ 3.3 h/head),
30-150K Gaussians/step (= cropped coverage x 4 views). val@10 == val@20 (18.55 dB) is the 500-step LR warm-up: the
zero-initialised head has not moved yet, so 18.55 dB is the UNTRAINED "splat the backbone's points" baseline — the
anchor training must beat. A 39 dB training step = an object filling little of the frame (full-image PSNR; LVSM's
protocol too). Nothing to fix. (v146: manual white compositing instead of gsplat's `backgrounds` argument.)

## Rows (decided): VGGT-Ω, VGGT, π³, STream3R — each raw and +SMR (8 rows)
One window = no junction graph, so +SMR+PGO does not exist for NVS (it would equal +SMR); the caption says so once.
Table 1 keeps its three rows because they sit beside the camera-pose columns where all three apply. π³ gets a
measured +SMR row (expected identity) rather than an asserted one.

## Launch (4 heads, 3 GPUs: GPU0 runs vggt_omega then stream3r; GPU1 vggt; GPU2 pi3; ≈ 6.6 h wall)
```bash
cd ~/smr && unzip -o smr_updates147.zip
DATA=~/data/nvs/objaverse_c70 bash scripts/nvs/train_all.sh
tail -n 2 outputs/nvs/*/train.log            # a minute later: three heads stepping
```
When all four have `done:` in train.log (or after ckpt_best exists for each):
```bash
GSO=~/data/nvs/gso_c70 bash scripts/nvs/eval_all.sh     # 8 rows: reads=1 and reads=4 per backbone; ~40 min
tar czf outputs/nvs/nvs_results.tgz outputs/nvs/*/gso_reads*.jsonl outputs/nvs/*/val.csv outputs/nvs/*/metrics.csv outputs/nvs/*/train.log
```
Send `nvs_results.tgz`; I fill Table 3, write Sec. 5.4's result sentences, and return the Overleaf zip.

## Predictions (recorded before training)
- P43 (standing): VGGT +SMR read: +0.3 to +1.0 dB PSNR over raw.
- P47: VGGT raw 24-27 dB (trained head; the untrained baseline is 18.55).
- P48: π³ raw ≥ VGGT raw. P52: VGGT-Ω raw ≥ VGGT raw. P53: STream3R +SMR ≤ raw (orderings not commensurable, as on
  ETH3D). P54: π³ +SMR == raw to within 0.05 dB (identity measured).
