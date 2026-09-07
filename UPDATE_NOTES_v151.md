# smr_updates151 — qualitative figures: tooling + one-shot recipe

## What the figures should show (and which VGGT-SLAM figures they correspond to)
1. **Camera pose vs length (CO3D, N=200)** — `scripts/fig_qualitative.py`, three panels from one pilot_a run:
   (a) pairwise relative-pose error matrices (deg, the AUC@30 criterion) for raw vs +SMR on the same colour scale with
   window boundaries drawn: the seam bleed IS the off-diagonal blocks, and the memory's effect is where they fade;
   (b) top-down trajectories after Sim(3) alignment (GT, raw, +SMR, +SMR+PGO) with accepted closures as green arcs;
   (c) per-frame position error with closures marked. This is the figure VGGT's paper does not have and ours needs:
   it makes Sec. 2 visible.
2. **SLAM on 7-Scenes** — same script on office and pumpkin (the largest recoveries), plus `scripts/fig_recon_topdown.py`:
   top-down fused clouds under the raw junctions vs under +SMR with a zoom inset on the revisited region — ghosted
   geometry left, aligned right. That is the analogue of VGGT-SLAM's Fig. 1 (Sim(3) vs SL(4)); ours contrasts chained
   Sim(3) windows with the memory's verified closures. Their Figs. 7/8 (frusta coloured by submap) correspond to our
   panel (b) with windows; a direct Sim(3)-vs-SL(4)-vs-SMR qualitative would require running their code, which the
   table already compares numerically — not worth a GPU day.
3. **NVS** — `experiments/eval_gso.py --save-images` + `scripts/fig_nvs_strip.py`: rows = objects, columns =
   4 inputs | GT target | raw render | +SMR render, PSNR stamped on the renders.

## Tooling (all tested here on synthetic data)
- `scripts/fig_qualitative.py --est <save-est npz> --json <report> --out <prefix>`
- `scripts/fig_recon_topdown.py --raw a.ply --smr b.ply [--zoom X Y HALF] --out <prefix>`
- `experiments/eval_gso.py ... --save-images DIR --save-n 8`, then `scripts/fig_nvs_strip.py --dir DIR --objects ... --out <prefix>`
- `scripts/fig_all.sh` runs everything on GPU 0 (~30 min): 2 CO3D orbits, office + pumpkin (trajectories + clouds), 8 GSO objects.
  Edit the CO3D/SCENES globs at the top if your GT npz names differ (`ls data/gt/`).

## Run
```bash
cd ~/smr && unzip -o smr_updates151.zip && source .venv/bin/activate
ls data/gt/ | head -30                      # check the npz names, then:
CUDA_VISIBLE_DEVICES=0 bash scripts/fig_all.sh
ls outputs/figures/                         # qual_*.pdf, recon_*.pdf, nvs_vggt.pdf (+ .png previews)
tar czf outputs/figures.tgz outputs/figures
```
Send `figures.tgz` (or just look at the PNGs first). I will pick one CO3D panel + one 7-Scenes panel + the NVS strip
for a main-text Figure 3 (~0.4 page; I will make the room) and put the rest in an appendix gallery.
For the recon zoom: run once without --zoom, read off the (x,y) of the revisited region from the PNG, then rerun
`fig_recon_topdown.py --zoom X Y 0.5`.
