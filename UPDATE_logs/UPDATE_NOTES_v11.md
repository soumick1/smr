# Update v1.1 -- sweep + pi^3 results recorded; three small fixes

Doc (smr_overleaf.zip, recompiled clean): Sec. "Measured status" gains
the full LLFF table (8 VGGT scenes + pi^3 room row), the two-curve
reference-drift figure (VGGT rises to ~29 deg; pi^3 flat 1-4 deg --
their equivariance claim confirmed by our probe), the trex held-out
figure, and the cross-backbone paragraph.  Images ship inside the zip
under images/ -- keep using that folder for future figures.

Code fixes (apply zip as usual):
* GT-path sanity figure now backbone-tagged (pi^3's run had overwritten
  vggt_sanity.png -- naming miss #2, both now closed).
* llff_pose_check: legend uses the actual backbone name, and the four
  headline numbers (centre median/mean, rel-rot median, abs-rot median)
  are archived to outputs/reports/llff_<backbone>_<scene>.json so they
  are never console-only again.

Known duplicate in your results: tier3_vggt_images_8.json is the
pre-fix trex run under the old collided name -- identical to
tier3_vggt_trex.json; delete it.
