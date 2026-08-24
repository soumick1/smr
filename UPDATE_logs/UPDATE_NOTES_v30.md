# Update v3.0 -- two K=1 bugs fixed (one you hit, one you would have)

1. YOUR CRASH: src/smr/backbones/vggt.py used images.squeeze(0) on the
   (K,3,H,W) loader tensor -- a no-op for K>=2 (every prior run), but it
   eats the view dimension exactly at K=1.  Fixed with a dim guard.
2. THE SILENT ONE (found tracing #1): the residue-decode envelope is
   +-1.6 scene units, and the K=1 camera sits at the origin -- so orbit
   targets beyond ~110 deg azimuth landed OUTSIDE the envelope and the
   chained decode would wrap, silently corrupting the trajectory at
   exactly the big sweeps.  The demo now pre-scales the scene for the
   worst pose it will visit (same Sim(3) trick as fit_decode_window;
   recorded as envelope_rescale in report.json), so every azimuth up to
   +-150 deg decodes cleanly.

Relaunch: Ctrl-C, then
    CUDA_VISIBLE_DEVICES=0 python app/gradio_demo.py
Rehearse once end-to-end (az +60 and az +140, both paths) before the
meeting.
