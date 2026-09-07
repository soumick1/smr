# smr_updates153 — publication-style reconstruction renders; fixes from the first figure pass

## What the first pass showed
- Pose figures for two CO3D orbits and the office trajectory panel rendered; the fused-cloud step failed because
  points_suite --pilot still needs --gt (fixed in the recipe).
- Window boundaries: my figure script treated the saved chunks as GT frame ids; they are keyframe POSITIONS -> only
  one boundary was drawn and the cross-window mask was wrong. Fixed (fig_qualitative.py).
- **office seq01 at stride 10 has 0 accepted closures** (raw 0.029 -> +SMR 0.027): a single 100-keyframe walk with no
  revisit is the wrong material for a memory figure. Table 2's office row (0.047 -> 0.026) and pumpkin (0.161 ->
  0.061) came from a revisit-rich configuration. The recipe now defaults to the `_all` files; SUFFIX=seq01 / STRIDE=N
  override. **Please tell me the Table-2 configuration**: `ls outputs/reports/ | grep -i "office\|pumpkin"` and, for one
  file, `python -c "import json,sys; d=json.load(open(sys.argv[1])); print({k:d[k] for k in d if k not in ('rows','events')})" <file>`.

## New: scripts/render_cloud.py (the VGGT-SLAM look, no display, no 3D library)
Z-buffered coloured point splats (colours now written into the fused PLYs by points_suite; height colouring when
absent), camera frusta coloured by window, accepted closures as green links, an orange zoom inset shown at the same
image location in every panel, raw and +SMR rendered from ONE camera fitted to the first cloud, 2x supersampled.
    python scripts/render_cloud.py --ply RAW.ply SMR.ply --labels "raw" "+SMR" --est est.npz --rows chained smr --json rep.json \
        --view oblique --azimuth 35 --elev 38 [--zoom X Y HALF] --out outputs/figures/recon_<scene>
Workflow for the inset: render once without --zoom, look at the PNG, pick the ghosted region (image fractions), rerun
with --zoom X Y 0.12. Try --azimuth -30 / --view top if the default viewpoint hides the revisit.

## Run
```bash
cd ~/smr && unzip -o smr_updates153.zip
CUDA_VISIBLE_DEVICES=0 bash scripts/fig_all.sh              # `_all` scenes, stride 10 (override: SUFFIX=... STRIDE=...)
tar czf outputs/figures.tgz outputs/figures && ls outputs/figures
```
Send figures.tgz (PNGs are enough to choose viewpoints and insets); I compose Fig. 3 + the appendix gallery from them.
