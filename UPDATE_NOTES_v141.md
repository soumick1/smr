# smr_updates141 — dry run: Objaverse renders pass; GSO textures fixed

## Scored
- **P44 hit (better than predicted):** 20/20 Objaverse objects rendered, 0 failed, 0.14–0.55 s/view (most 0.15–0.20;
  the slow ones are import-bound: a 123K-vertex, 1,500-node GLB took 5.6 s to import). Import + normalise 0.02–5.6 s.
  Revised full-run estimate (P46'): 20K objects x 16 views on 3 GPUs ≈ **6–7 h**.
- **P45 hit:** 20 GLBs in 18 s (median 1.2 MB); 5 GSO models in 17 s (listing via `/1.0/models?q=collections:...`,
  299 models returned by the API — GSO has 1,030; see below).
- bpy is **5.2** (venv Python 3.13): glTF import, OBJ import (`wm.obj_import`), OptiX all work; `World.use_nodes`
  deprecation warning is harmless.

## What failed: GSO textures
GSO's `model.mtl` references `map_Kd texture.png` by bare name while the archive stores it at
`materials/textures/texture.png`; Blender resolves MTL paths relative to `meshes/`, so every GSO object rendered
untextured (the "Cannot load image file ... meshes/texture.png" warnings). Fixes:
- `scripts/nvs/gso_fetch.py` links every file under `**/textures/` next to the .obj after extraction (also for
  models already on disk: re-running the fetcher repairs them, no re-download).
- `scripts/nvs/render_bpy.py` relinks any image whose file is missing to a same-named file under the model folder
  (generic; counts appear in meta.json / the log as "textures relinked" or "MISSING textures").
- glTF importer INFO spam silenced (`loglevel=30`; 1,500 lines per complex object otherwise).

## GSO count: 299 vs 1,030
The collection query returned 299 models. Either the API caps the search or the collection listing differs from the
full GoogleResearch upload (1,030 per the GSO paper). Check with the owner route by names: after the dry run, run
`python scripts/nvs/gso_fetch.py --index data/nvs/gso_index.json` and read the "listing via ..." line; if it stays at
299 we evaluate on those 299 (stated in the caption) unless the names-file route finds more — tell me the number.

## Redo the GSO dry run (~2 min), then LOOK
```bash
cd ~/smr && unzip -o smr_updates141.zip && source .venv/bin/activate && python tests/test_render_bpy_mock.py
python scripts/nvs/gso_fetch.py --limit 5 --index data/nvs/gso_index_dry.json     # repairs texture links in place
rm -rf ~/data/nvs/dry_gso
CUDA_VISIBLE_DEVICES=0 python scripts/nvs/render_bpy.py --list data/nvs/gso_index_dry.json --out ~/data/nvs/dry_gso --protocol gso --obj-up Z --device OPTIX 2>&1 | grep -v "Saved:"
```
Expect "N texture(s) relinked" or no texture warnings. Then open `~/data/nvs/dry_gso/2_of_Jenga_Classic_Game/000.png`
(input view: textured box seen from 20 deg above, white background) and one objaverse render, e.g.
`~/data/nvs/dry_objaverse/6b6928f3094f41b38ab9941d0cc8f618/000.png` (a wine bottle: should be upright). If either
object is lying on its side, tell me which.

## Then the full run (unchanged)
```bash
python scripts/nvs/objaverse_fetch.py --n 20000            # ~30-40 GB, < 1 h
python scripts/nvs/gso_fetch.py                             # all listed models
bash scripts/nvs/render_all.sh                              # 3 GPUs; ~6-7 h; GSO after shard 0
```
