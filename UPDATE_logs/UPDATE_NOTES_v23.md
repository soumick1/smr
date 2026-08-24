# Update v2.3 -- measurable novelty

Every new harvested pair now stores novelty_rot / novelty_pos (pose gap
to the NEAREST bound context view) + the bound context ids, and the
harvester prints them live.  Eval sample figures now carry the source
pair's filename in the title, so every qualitative panel is traceable
to its scene / holdout / context class.

NEW scripts/pair_stats.py: corpus report -- coverage and novelty
histograms per context class:
    python scripts/pair_stats.py data/completion_co3d data/completion
(Pairs harvested before this update lack the novelty fields and show
"n/a"; coverage still reports.  No re-harvest needed -- cycle pairs
will carry the fields.)

Interpretation guide: call/c04 pairs are small-baseline NVS (novelty ~
adjacent view spacing: ~2.5 deg median on LLFF sweeps, ~11-12 deg on
32-frame orbits -- both derivable from the V3 thresholds, which record
2x median spacing).  c01 novelty is better read as 1-coverage: one
bound view makes every parallax-revealed surface a hole.  The orbit
demo's 115 deg is pose-TRAVERSAL length; rendering at arrival is
always local recall (k=2 nearest, ~11 deg) + completion.
