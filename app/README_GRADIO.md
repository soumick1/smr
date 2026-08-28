# SMR interactive demo — gradio-only runbook

Everything needed to run `app/gradio_demo.py` on the lab server and nothing
else. The demo takes one photo, lifts it to geometry with a frozen backbone,
binds it into the scaffold memory, physically drives the attractor to a
target viewpoint you pick, and renders the sweep next to what the bare
backbone can do alone.

## 1. Environment (once)

    cd ~/smr && source .venv/bin/activate
    pip install gradio plotly            # the only demo-specific packages

The demo needs the completion checkpoint for disocclusion filling:

    ls outputs/logs/completion/co3d_v1/ckpt_best.pt   # must exist
    # if missing: unzip smr_ckpt_completion.zip into outputs/logs/completion/co3d_v1/

## 2. Launch

    CUDA_VISIBLE_DEVICES=0 python app/gradio_demo.py            # http://localhost:7860
    CUDA_VISIBLE_DEVICES=0 python app/gradio_demo.py --share    # public 72 h link
    # over ssh instead of --share:
    #   ssh -p 22 -L 7860:localhost:7860 soumick@10.97.144.63   # then open localhost:7860

## 3. Backbones in the demo

| choice      | first-load time | peak GPU  | notes                                   |
|-------------|-----------------|-----------|-----------------------------------------|
| vggt        | ~20 s           | ~9 GB     | default                                  |
| vggt_omega  | ~20 s           | ~6 GB     | best geometry of the four (paper Tab. 1) |
| pi3         | ~15 s           | ~10 GB    |                                          |
| fast3r      | ~25 s           | ~9 GB     | weakest single-image geometry; the demo shows why the tables rank it low |

Models are cached after first use (one at a time; switching frees the old
one). The demo is single-image, so DUSt3R-family models (pairwise, 50-75 s a
pass) are deliberately not offered.

## 4. Controls, in the order a demo runs

1. **photo** — any indoor/object photo; it is centre-cropped to the
   backbone's resolution.
2. **azimuth / elevation / distance** — the target viewpoint on the orbit
   around the subject (the pose preview updates live; distance is a factor
   of the subject distance, so 1.0 orbits at constant range).
3. **speed** — slow / normal / fast: frames per second of scaffold driving
   (2/4/7 steps per frame); slow is best on a projector.
4. **completion head** — on: disocclusions filled by the trained head;
   off: raw surfel splat with holes (useful to show what memory alone gives).
5. **run** — renders the sweep (left: SMR driving there; right: the
   backbone's own reprojection at the *given* target pose, no path). The
   ring/grid animation under the sweep is the scaffold state, live.

## 5. What to say while it runs (30-second script)

The backbone turns the photo into surfels and a camera. The memory binds
them at a scaffold state. Driving the attractor moves the state along the
requested arc; at every step the memory is read out and rendered -- the
backbone is never called again. The right panel is the backbone teleported
to the target with no memory: same pixels, no path, holes where it never
looked.

## 6. Troubleshooting

* **"CUDA out of memory" on switch** — switch backbones only between runs;
  the old model frees on the next click. Worst case: restart the app.
* **Port already in use** — `pkill -f gradio_demo` or launch with
  `GRADIO_SERVER_PORT=7861`.
* **--share link dead** — links expire after 72 h; relaunch.
* **Slow first render** — DINO + backbone load once per process; subsequent
  runs are seconds.
* **Completion checkbox greyed** — checkpoint missing; see §1.

Nothing in the demo writes to `outputs/reports/`; it cannot disturb paper
runs, and it shares the GPU politely (one pass at a time).
