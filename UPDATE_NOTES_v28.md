# Update v2.8 -- the professor demo (Gradio), full handholding

## What it is
app/gradio_demo.py -- a web app on your server.  Your professor (or
you) uploads ONE photo, picks a target viewpoint on a live 3-D
selector (azimuth / elevation / distance sliders drive a plotly scene
showing reference camera, target camera, and the mental path), picks
the backbone (vggt / pi3), and presses "Drive the bumps."  The app then
binds the photo (K=1), physically drives the scaffold to the target,
and produces:
  * a master animation: [SMR completed render | backbone-reprojection-
    only | top-down path] over [yaw/pitch/roll rings + all 3 grid
    modules] -- every panel in perfect sync because they come from the
    same snapshots.  "Slow" speed makes the integration obvious.
  * a clean side-by-side view-comparison GIF
  * a summary (steps, arrival error, completion status) + report.json
  * one-click "download everything" zip  ·  "Start afresh" wipes state

## Install + launch (5 minutes, first time)
    cd ~/smr && source .venv/bin/activate
    pip install gradio plotly
    CUDA_VISIBLE_DEVICES=0 python app/gradio_demo.py
You'll see:  Running on local URL: http://0.0.0.0:7860
(A Gradio 6 deprecation warning about theme/css is expected; ignore.)

## Viewing it
  * On the server's browser: http://localhost:7860
  * From YOUR laptop (recommended):
        ssh -p 22 -L 7860:localhost:7860 soumick@10.97.144.63
    then open http://localhost:7860 in your laptop browser.
  * For your professor WITHOUT any network setup:
        CUDA_VISIBLE_DEVICES=0 python app/gradio_demo.py --share
    prints a public https://xxxx.gradio.live link, valid ~72 h.
    (Anyone with the link can use your GPU -- share only with her,
    close with Ctrl-C when done.)
Run it inside screen so the link survives disconnects:
    screen -S demo
    CUDA_VISIBLE_DEVICES=0 python app/gradio_demo.py --share

## Runtimes (per press of "Drive the bumps")
  backbone forward ~10-20 s (model loads each run) + drive 2-5 s +
  sweep render 25-50 s (slow mode) => ~45-80 s total.  The progress
  bar narrates each phase.

## Suggested 3-minute script for the meeting
  1. Upload a photo of anything on her desk.  az=+60, slow, vggt.
  2. While it renders: point at the 3-D selector -- "this is the pose
     we'll imagine from; the orange arc is the path the bumps take."
  3. When the GIF plays: point at the bottom row -- "these are the
     actual attractor networks; the dashed line is the decoded heading;
     rendering happens at the pose read off these bumps."
  4. Right panel: "this is all the backbone can do alone."
  5. Re-run with completion UNCHECKED -- the black holes are the honest
     disocclusions; the checkbox is the one trained component.
  6. az=+120: bigger sweep, bigger holes, head fills more -- the K=1
     limit of the same machinery that passed 25/25 at scale.

## Troubleshooting
  * "Upload a photo first" -- the run button needs an image.
  * Port busy: --port 7861 (and forward that port instead).
  * pi3 prints a RoPE slow-path warning: normal.
  * Any pipeline error surfaces in-app with its real type + message.
