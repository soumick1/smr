# Update v0.6 -- read your four runs correctly, then real photos properly

## THE MILESTONE IS GREEN (your run 2, synthetic frames + VGGT)
V1 0.0029 rad | V3 10/11 (median reloc pose err 0.0003) | V5a 0.0218 |
V5b 0.0222 -- 2.2% end-to-end vs the true world through bind -> place ->
recall -> splat on VGGT's ESTIMATED geometry.  Backbone swap validated.
(Your console has it; the JSON was overwritten by the later photo run --
report names now include the frames tag so this cannot recur.)

## Sanity run: one bad view, not a systemic problem
Per-pair printout localises it: pairs 5-6 and 6-7 at 0.46, all others
0.009-0.047; view 6 (the 180-degree antipode) at 28 deg while the rest
sit <=8.  Means are outlier-dragged; the script now prints medians and
names the worst view.

## Photo runs: input-contract violation, correctly detected
The folder contained unrelated stock photos of DIFFERENT scenes (view 0
is a poppy field) at four aspect ratios.  VGGT's contract is K views of
ONE static scene; self-consistency 3.9-98.8 is the metric doing its job.
The wrap (V1 pos 2.38) and NaN were unguarded edges, now fixed:
* fit_decode_window(): camera centres outside +-max(period)/2 trigger a
  loud [NOTE] and a Sim(3) rescale into the window (verified: 5x-blown
  centres -> 1.600; well-posed scenes untouched, synthetic regression
  bit-identical all-pass).
* V5 with <200 overlapping pixels now FAILS with the message "are these
  frames views of ONE static scene?" instead of NaN.

## Capture protocol (unchanged, now enforced by diagnostics)
10-15 photos of ONE corner of the lab, slow arc, 30-50% overlap between
consecutive shots, same phone orientation for all, textured content, no
motion blur.  Then:
    python scripts/vggt_sanity.py --frames <dir>
    python experiments/run_tier3_vggt.py --backbone vggt --frames <dir>
