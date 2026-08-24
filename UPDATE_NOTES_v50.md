# Update v5.0 -- 7/10 -> the last three

vggt_omega PASSED (gated download + adapter both work) and monst3r's
weights are on disk.  Three issues left, all dependency-level.

## streamvggt: No module named 'transformers'
Their model file imports it at module level.  ONE command:

    pip install --no-deps transformers tokenizers safetensors regex

--no-deps protects torch and numpy; every other transformers dependency
(filelock, huggingface-hub, pyyaml, requests, tqdm, packaging, numpy) is
already in the venv.  setup_backbones.sh now does this for the streamvggt
target.

## monst3r: No module named 'evo'  -- fixed in code, nothing to install
`dust3r/utils/vo_eval.py` imports `evo` (a trajectory-EVALUATION package)
at module level; nothing on the inference path uses it.  Now stubbed, via a
helper I should have factored out two backbones ago:

    import_with_optional_stubs(loader, optional, label)

It retries the import while permissively stubbing ONLY the named modules
and re-raises anything else.  fast3r now uses it too, and there are tests
for both directions (listed module -> stubbed and retried; unlisted module
-> re-raised).  Allowlists stay tiny and explicit:
    fast3r  : pl_bolts, open3d, wandb
    monst3r : evo, open3d, wandb

## cut3r: two problems, and my advice is to drop it
1. `missing module(s): ['src.dust3r.model']` -- probing that path executed
   their whole model module.  Lightened to `src.dust3r`, which only proves
   the repo is importable.
2. Its weights are quota-blocked: "Too many users have viewed or downloaded
   this file recently... may take up to 24 hours".  Nothing we can do but
   wait, or grab it via a browser.

You will have NINE backbones without it.  cut3r also has the least
convenient API in the set (a demo script at repo root, a `src.` package
layout).  Retry the download tomorrow if you want the tenth row; do not
spend today on it.

## Run
    pip install --no-deps transformers tokenizers safetensors regex
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r \
                    streamvggt monst3r vggt_omega --max-views 4

Tests: 109 passed, 10 GPU-tier skipped.
