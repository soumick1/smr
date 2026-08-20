# Update v0.2 -- apply over the repo root

Unzip `smr_updates.zip` over `~/smr`, review with `git diff`, then:
    git add -A && git commit -m "v0.2: T6 leak criterion, detect->correct (I4), VGGT wiring"
Re-run: `python experiments/run_all.py` -> expect 25/25
(T6 is now two checks; I4 adds two; tier1=12, tier2=6, tier3=7).

## Changed / added
* experiments/run_tier1_dynamics.py -- T6 holonomy measured as SPIN-
  ATTRIBUTABLE displacement vs a matched idle window (old criterion was
  accidentally duration-dependent: intrinsic CANN drift ~0.0022 units/t
  * 60 t = the 0.133 your full run saw).  Leak measured = 0.0; drift rate
  reported as its own diagnostic check; figure updated.
* src/smr/pipeline/rotate.py -- `coherent_reanchor()`: novelty-gated
  detect->correct.  Position via least-squares phase fit (`_ls_position`),
  which is EXACTLY null on incoherent noise (per-axis block structure of
  C); re-places every module; wired into `rotate()` (`correct=True`,
  `novelty_tau=0.2`).  `settle_fiber` now defaults False (detector role).
* src/smr/pipeline/__init__.py -- exports `coherent_reanchor`.
* experiments/run_tier3_integration.py -- I4 added: incoherent injection
  (T11-validated norm 1.8) -> novelty 0.38 -> corrected to 0.043 with
  position restored to 0.0084; coherent control stays below the gate and
  correction is a verified no-op.  Figure I4_detect_correct.png.
* tests/test_pipeline.py -- reanchor no-op on clean state.
* src/smr/backbones/vggt.py -- full adapter (camera-from-world ->
  world-from-camera inversion, per-scene median-depth normalisation,
  confidence mask); every repo-specific call marked VERIFY-ON-SERVER.
* scripts/export_synthetic_frames.py, scripts/vggt_sanity.py -- the
  week-3 backbone-swap check as a two-command run.
* README.md -- VGGT wiring section.

## Document (smr_overleaf.zip, recompiled: 41 pp, 0 errors/undef refs)
* T6 row: spin-attributable-leak operationalisation + drift diagnostic.
* T11 row: selective DETECTABILITY (>50x criterion; measured ~1e6).
* New Remark rem:detector after Cor. base-blind: loop = detector with
  continuous codes; correction = novelty-gated coherent re-anchor (LS fit
  null on eps_perp); Cor. base-blind unchanged.
* T14 row: generalisation over COMBINATIONS + contiguous negative control.
* I4 row: detect->correct protocol; settle-as-consistency subsection now
  routes through the re-anchor operator.
