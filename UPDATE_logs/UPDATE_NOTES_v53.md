# Update v5.3 -- cut3r: the stale sys.path entry (9/10 -> 10/10)

monst3r now PASSES.  cut3r's last error names the culprit in its own path:

    cannot import name 'CrocoConfig' from 'models.croco'
    (/home/soumick/smr/third_party/DUST3R/croco/models/croco.py)

It was loading **DUSt3R's** croco.  Why: dust3r/model.py bootstraps itself
by inserting its vendored croco directory into sys.path at import time, and
that entry SURVIVES our sys.modules purge -- so CUT3R's
`from models.croco import CroCoNet, CrocoConfig` resolved to DUSt3R's copy,
which has no CrocoConfig.  (CUT3R vendors its own croco at
CUT3R/src/croco, which does define it.)

Two fixes:
  * `isolate_third_party()` now also REMOVES third_party sys.path entries
    that do not belong to the repos being prioritised -- purging modules was
    never sufficient while a stale path entry remained ahead;
  * cut3r declares ("CUT3R/src/croco", "CUT3R/src", "CUT3R") so its own
    croco leads, and purges the "models" prefix too.

This was the deepest instance of a pattern worth stating in the paper: six
of these ten repos vendor forks of the same two packages (dust3r, croco)
and self-bootstrap sys.path, so loading several in one process requires
explicit namespace isolation.  For the evaluation harness, prefer one
process per backbone.

    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r \
                    streamvggt monst3r vggt_omega cut3r --max-views 4

Tests: 114 passed, 10 GPU-tier skipped.
