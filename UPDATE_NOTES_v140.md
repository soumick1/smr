# smr_updates140 — GSO fetcher: listing endpoint fixed (ladder + names-file fallback)

## What failed
`GET /1.0/GoogleResearch/models?page=1&per_page=100` -> HTTP 404. GSO is a *collection* ("Scanned Objects by Google
Research", owner GoogleResearch) and Fuel's routes for listing have moved over time. The download form itself
(`<model URL>.zip`, documented by Fuel) was not exercised yet.

## Fix
`scripts/nvs/gso_fetch.py` tries, in order, printing each attempt's HTTP status:
1. `/1.0/models?page=N&per_page=100&q=collections:Scanned%20Objects%20by%20Google%20Research` (the route the widely used
   GSO download snippet relies on);
2. `/1.0/GoogleResearch/collections/<name>/models?page=N&per_page=100`;
3. `/1.0/GoogleResearch/models?...` and the lowercase-owner variant.
If all fail: `--names-file gso_names.txt` (one model name per line; names are visible on the collection page) skips
listing and downloads directly. Downloads try `<owner>/models/<name>.zip` then `<owner>/models/<name>/1/<name>.zip`;
bad archives / missing .obj are reported and skipped; resume-safe.

## Continue the dry run
```bash
cd ~/smr && unzip -o smr_updates140.zip && source .venv/bin/activate
python scripts/nvs/gso_fetch.py --limit 5 --index data/nvs/gso_index_dry.json      # prints which listing route worked
CUDA_VISIBLE_DEVICES=0 python scripts/nvs/render_bpy.py --list data/nvs/objaverse_index_dry.json --out ~/data/nvs/dry_objaverse --protocol train --n-views 8 --device OPTIX
CUDA_VISIBLE_DEVICES=0 python scripts/nvs/render_bpy.py --list data/nvs/gso_index_dry.json --out ~/data/nvs/dry_gso --protocol gso --obj-up Z --device OPTIX
grep "s/view" ~/data/nvs/dry_*/render_shard0.log | tail -25
```
Then look at `~/data/nvs/dry_gso/<first>/000.png` (input view from 20 deg above, white background) and one objaverse render.
Objaverse note: LVIS objects are small (median 1.2 MB; 20 in 18 s) -> the full 20K download is ~25-40 GB, < 1 h.
Predictions P44-P46 unchanged.
