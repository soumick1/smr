#!/bin/bash
# The README / project-page demo on the real office sequence (VGGT-Omega run of Fig. 3).  ~15 min on the server (CPU).
#   bash scripts/make_demo.sh [--kf-range 0 150]
# Writes docs/assets/demo.mp4 (project page), docs/assets/demo.gif (README: 15 fps, 128 colours, ~13 MB) and demo_poster.png.
cd "$(dirname "$0")/.."; source .venv/bin/activate
GT=${GT:-data/gt/7scenes_office_seq01.npz}; REP=${REP:-outputs/reports/fig_7scenes_office_seq01.json}
python scripts/demo_video.py --gt $GT --report $REP --out docs/assets/demo_full --frames-per-kf 2 --fps 20 --width 1280 --title "SMoRe on a real sequence" "$@"
mv -f docs/assets/demo_full.mp4 docs/assets/demo.mp4
# README-sized GIF: keep 3 of every 4 frames, one 128-colour palette, only changed rectangles re-encoded (61 MB -> 13 MB)
ffmpeg -y -loglevel error -i docs/assets/demo_full.gif \
  -filter_complex "[0:v]fps=15,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  -loop 0 docs/assets/demo.gif
python - <<'PY'
from PIL import Image
im = Image.open("docs/assets/demo.gif"); im.seek(0); im.convert("RGB").save("docs/assets/demo_poster.png")
print("poster written; gif", im.n_frames, "frames", im.size)
PY
rm -rf docs/assets/demo_frames docs/assets/demo_full.gif
ls -la docs/assets/demo.gif docs/assets/demo.mp4 | awk '{printf "%s %.1f MB\n", $9, $5/1e6}'
