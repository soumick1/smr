# smr_updates193 — choosing and rendering the NVS qualitative examples (Fig. 4)

Three scripts, no code patches:
* `scripts/nvs_pick_examples.py`: ranks the 1,033 GSO objects by the raw -> +SMR gain (PSNR, LPIPS reduction, or both),
  with filters so the +SMR render is good and the raw render bad; prints the gain distribution and a `--ids` line.
* `scripts/nvs_render_objects.py`: renders chosen objects with reads 1 and reads 4 through the same head (mirrors
  eval_gso.py), saves inputs / targets / both predictions for all ten targets, and prints per object the target index with
  the largest PSNR gain and the largest LPIPS reduction (`per_target.json`, `summary.json`). ~1 min per object.
* `scripts/fig_nvs_strip.py` (updated): `--targets` one index per object, `--error-maps` adds |raw − target| and
  |+SMR − target| columns on a shared colour scale, `--lpips` adds LPIPS to the labels.

```bash
unzip -o smr_updates193.zip
# 1. candidates (seconds); the +SMR render must reach 20 dB, the raw render must be below 22 dB, both criteria matter
python scripts/nvs_pick_examples.py --raw outputs/nvs/vggt/gso_reads1.jsonl --smr outputs/nvs/vggt/gso_reads4.jsonl --rank combined --min-smr-psnr 20 --max-raw-psnr 22 --top 12
# 2. render the top 12 with both reads (~12 min on one GPU); paste the --ids line the selector printed
CUDA_VISIBLE_DEVICES=0 python scripts/nvs_render_objects.py --backbone vggt --ckpt outputs/nvs/vggt/ckpt_best.pt --gso ~/data/nvs/gso --out outputs/nvs/vggt/images_pick --ids <ids from step 1>
# 3. strip of all 12 with each object's best target and error maps; pick 3 with your professor, then rerun with those 3
python scripts/fig_nvs_strip.py --dir outputs/nvs/vggt/images_pick --objects <ids> --targets <best_psnr_target per object from step 2> --error-maps --lpips --out outputs/figures/nvs_vggt_candidates
```
Send back the selector table, the renderer's per-object lines, and `outputs/figures/nvs_vggt_candidates.png`.
