# smr_updates144 — after the render: verify, smoke-test the torch stack, train

## 1. Verify the renders (~10 min; the alpha check reads every image once)
```bash
cd ~/smr && unzip -o smr_updates144.zip && source .venv/bin/activate
tail -n 1 outputs/reports/nvs_render_gpu*.log outputs/reports/nvs_render_gso.log      # "shard k: rendered X, failed Y"
python scripts/nvs/check_renders.py --root ~/data/nvs/objaverse --views 16
python scripts/nvs/check_renders.py --root ~/data/nvs/gso --views 14
du -sh ~/data/nvs/objaverse ~/data/nvs/gso; df -h ~ | tail -1
```
`check_renders.py` counts objects, FAILED markers, missing views, non-RGBA images, and the object coverage per
image; it flags EMPTY objects (< 0.5 % coverage: nothing visible) and SATURATED ones (> 85 %: the frame is full,
i.e. normalisation did not do what it should) and writes `<root>/exclude.txt`, which the trainer and the GSO
evaluator now honour. Expected: coverage median 20-45 %, a few percent EMPTY (flat/degenerate assets), ~0 SATURATED,
GSO with 1,033 objects and (almost) nothing excluded. Paste the two summaries. If GSO has EMPTY objects, look at one
of them (`ls ~/data/nvs/gso/<id>/`, open 000.png) before we decide anything.
**Predictions:** P49 objaverse FAILED + EMPTY + SATURATED <= 4 % of 20K; P50 GSO excluded <= 5 objects.

## 2. Smoke-test the torch stack (GPU 0, ~5 min) — unchanged from v142 §4; run it if not yet done
```bash
pip install gsplat lpips torchmetrics && python -c "import gsplat, lpips; print('gsplat', gsplat.__version__)"
python tests/test_nvs_geometry.py
CUDA_VISIBLE_DEVICES=0 python experiments/train_nvs.py --backbone vggt --data ~/data/nvs/objaverse --out outputs/nvs/smoke --smoke 2>&1 | tail -30
```
(The real renders exist now, so the smoke run uses them directly; no dry re-render needed.) Expect 20 steps,
"gaussians" ~50-150K per step, a falling loss, ~0.3-0.8 s/step, one val line. Paste the tail — the first
torch/gsplat error is what I need to see.

## 3. Train (after the smoke run passes; ~3-4 h per head, one GPU each)
```bash
bash scripts/nvs/train_all.sh          # vggt / pi3 / mast3r -> outputs/nvs/<bb>/{train.log,metrics.csv,val.csv,ckpt_*}
tail -n 3 outputs/nvs/*/train.log
```
Then `bash scripts/nvs/eval_all.sh` for the Table 3 rows (raw and read). Keep the GLBs (155 GB) until one head has
trained end to end; then `rm -r ~/data/nvs/objaverse_glb` if disk is needed.
