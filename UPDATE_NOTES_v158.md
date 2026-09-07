# smr_updates158 — reconstruction figures that look like the published ones (fig_recon.py)

## Root cause of the "childish" renders
`properties ['x','y','z']`: the rebuilt clouds STILL had no colours -- the colour sampler in points_suite failed
silently and fell back to xyz, so the renderer painted height with viridis. Their figure is 90 % colour.
Second, my viewpoint heuristic (up = least-variance axis) picked the wrong axis for the kitchen, and the frusta were
tiny and empty. None of this was the reconstruction: the clouds are 51 M coloured-able points per row.

## fig_recon.py (replaces render_cloud.py for figures; developed on the real pumpkin pack)
- **Colours by reprojection**: every point is projected into the row's keyframes (est poses + intrinsics) and takes
  the colour of the nearest camera that sees it -- the point came from one of those depth maps. No dependency on the
  fusion code. Intrinsics: --K, else the GT npz (CO3D), else 7-Scenes' 585/585/320/240.
- **Viewpoint**: up = the cameras' own up; eye behind and above the trajectory looking at the observed volume
  (defaults tuned: azimuth -12, elev 34, dist 1.9 x scene radius, fov 50); --crop drops only far-flung fragments.
- **Frusta with images** (every ~8th keyframe, coloured by window, keyframe image warped onto the far plane),
  closures as green links, orange zoom inset at the same place in every panel, raw and +SMR from one camera.
- Preview from your pack: recon_pumpkin_preview.png (colours from 192-px thumbnails, 1.5 M points; the server run
  uses full-resolution images and 8 M points and will be sharper). The ghosting reads in the inset: raw's red
  cabinet fronts bleed through/above the grey counter top, +SMR's top is one clean surface.

## Run (CPU only for the 7-Scenes renders; the clouds exist)
```bash
cd ~/smr && unzip -o smr_updates158.zip
for s in pumpkin office; do n=fig_7scenes_${s}_seq01
  python scripts/fig_recon.py --est outputs/est/$n.npz --json outputs/reports/$n.json --ply outputs/points/${n}_chained/fused.ply outputs/points/${n}_smr/fused.ply \
     --rows chained smr --labels "raw (chained Sim(3) windows)" "+SMR" --zoom 0.40 0.27 0.13 --out outputs/figures/recon_$s
done
```
(~5 min each: 8 M points coloured against 200 full images; all figures 600 dpi, renders 2600x1800 px per panel.) Office may need its own --azimuth/--zoom: look at the PNG
without --zoom first, then set X Y (image fractions) on a ghosted region. Extra views: `--elev 72` (top-down) shows
the drift as doubled counter tops.

## CO3D orbits: the runs in figures.tgz are still the OLD 20-keyframe ones
`data/gt/co3d/*.npz` has 10 frames per sequence (that is the N=10 set); `co3d_full` was not in your printout. Check:
`python -c "import numpy as np,glob; [print(f, len(np.load(f,allow_pickle=True)['poses'])) for f in sorted(glob.glob('data/gt/co3d_full/*.npz'))[:3]]"`
If co3d_full has ~200 frames: `rm -rf outputs/est/fig_apple* outputs/reports/fig_apple* outputs/points/fig_apple* outputs/figures/*apple*`
then `NCO3D=2 SKIP_NVS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/fig_all.sh` (stride 1 -> 200 keyframes, 12 windows;
poses + two clouds + fig_recon render per orbit, ~12 min each). If it has fewer, tell me which directory fed Table 1.
