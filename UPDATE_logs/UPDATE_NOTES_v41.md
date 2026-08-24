# Update v4.1 -- vendored-fork shadowing + torchvision weight enums

## mvdust3r: it loaded the WRONG dust3r
The error named the file: third_party/**dust3r**/dust3r/model.py.  But
mvdust3r vendors its OWN dust3r fork (third_party/mvdust3r/dust3r), and
that fork is the only one defining AsymmetricCroCo3DStereoMultiView --
verified in the clone.

Root cause is subtler than a path order: check_backbones.py loads every
backbone IN ONE PROCESS, so by the time mvdust3r runs, plain dust3r is
already in sys.modules and `import dust3r` returns the cached module no
matter what sys.path says.

FIX: `isolate_third_party(("dust3r","croco"), "mvdust3r")` purges the
cached names and re-prioritises the vendored path, called before every
dust3r import in that adapter.  (mast3r survives because it ships its own
path_to_dust3r helper; must3r had the same latent hazard.)

Worth knowing for the evaluation harness: run each backbone in its own
PROCESS when you can.  Vendored forks of the same package are endemic in
this family, and process isolation is the only fully robust answer.

## fast3r: the ancient torchvision, a fourth time
    ImportError: cannot import name 'VGG16_Weights' from torchvision.models
    raised at torchmetrics/functional/image/dists.py:51
torchmetrics imports that enum (torchvision >= 0.13) at MODULE level, so
the whole Lightning stack fails to import.  `patch_torchvision_weight_enums()`
adds stub enums good enough for imports and default arguments of metrics we
never call, and prints a note when it fires.

## Please consider upgrading torchvision after this run
That is now FOUR separate failures from one cause (perceptual loss,
must3r's center_crop, must3r's resize, fast3r's torchmetrics):

    pip install --upgrade torchvision      # match torch 2.5.1 -> 0.20.x

It would very likely retire all four shims AND bring MUSt3R back. Do it
when you have 20 minutes to re-run check_backbones.py afterwards -- not
mid-experiment.

## Run
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r mvdust3r \
        --max-views 4

Tests: 90 passed, 8 GPU-tier skipped.
