# Update v2.9 -- demo revisions (all five requests)

1. Backbone panel is now the STATIC reprojection at the GIVEN target
   pose -- journey vs teleport, visible: SMR animates its way there,
   the backbone can only be told where "there" is.
2. Amari rings are now RINGS: polar plots, the bump as a radial
   swelling travelling around the circle, dashed spoke = decoded
   heading (signed degrees).
3. All three grid modules shown as phase sheets (module m/3 with live
   phi readout) -- unchanged, confirmed all present.
4. NEW: 3-D torus panel -- module 0's bump moving on the actual donut,
   activity as surface colour.
5. NEW: mental-path option [orbital | direct].  Orbital waypoints
   around the subject (object stays in view; step count grows with arc
   length -- the RT law in path form); direct is the chord.

Fixed: the 3-D target selector no longer resets your camera when you
move a slider (plotly uirevision), and its initial camera no longer
clips the lower arc.

## Relaunch (no reinstall needed)
    Ctrl-C the running app, then:
    CUDA_VISIBLE_DEVICES=0 python app/gradio_demo.py         # or --share
Rendering is a bit heavier per frame (10 panels incl. the 3-D torus):
slow mode ~60-90 s total per run; the progress bar narrates.
