# smr_updates197 — scaffold GIFs on a real 7-Scenes sequence with the loop closure shown

`scripts/anim_real_scene_gifs.py` (needs `scripts/anim_scaffold_gifs.py` next to it) drives the same Amari rings and tori
with the ground-truth trajectory of a 7-Scenes sequence and shows the real RGB keyframes. The accepted closures of the SMR
run are taken from the run's report and staged: once the camera has passed the frame a later window will retrieve, a
hollow "stored address" marker appears on every field (and on the top-down map); when the camera returns the bump lands
on it; at the accepting window the marker turns green, the remembered view is shown beside the current view, the closure
edge is drawn on the map and every panel reads "loop closure accepted". All GIFs: identical frame count, 50 ms per frame.

```bash
cd ~/smr && unzip -o smr_updates197.zip -d scripts/
# 1. which closures the pumpkin run has (prints windows, keyframe ranges and the retrieved keyframe; no rendering)
python scripts/anim_real_scene_gifs.py --gt data/gt/7scenes_pumpkin_seq01.npz --report outputs/reports/fig_7scenes_pumpkin_seq01.json --out outputs/anim_pumpkin --frames 0 0
# 2. the GIFs, first closure included, 300 dpi (~20 min); drop --kf-range for the whole sequence (~30 s of animation, ~30 min)
python scripts/anim_real_scene_gifs.py --gt data/gt/7scenes_pumpkin_seq01.npz --report outputs/reports/fig_7scenes_pumpkin_seq01.json \
    --out outputs/anim_pumpkin --kf-range 0 150 --frames-per-kf 3 --fps 20 --dpi 300
# office seq-01 (7 of 7 closures useful) is the other good candidate:
python scripts/anim_real_scene_gifs.py --gt data/gt/7scenes_office_seq01.npz --report outputs/reports/fig_7scenes_office_seq01.json --out outputs/anim_office --dpi 300
```
Options: `--closure-index 0` stages only the first closure; `--frames-per-kf 4` slows everything down (4 animation frames
per keyframe, 5 keyframes per second); `--unit 1.5` metres per scaffold unit (the module periods 2.4 / 3.2 / 4.0 are in
these units); `--only rings modules camera` renders a subset. The report must be the one of the run whose closures you
want to show (the `fig_7scenes_*` reports are the Fig. 3 runs, stride 5, 200 keyframes, W = 32 / O = 16); if the report
was produced with another stride or window, pass `--stride/--chunk/--overlap` to match, otherwise the keyframe indices
of the closures will not line up.
