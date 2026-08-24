# Update v3.2 -- checkpoint fetching, wired to the registry

## The answer to "how do I download each model's weights"

Verified from each repo's README (not memory), 2026-08-24:

| backbone | checkpoint | HF hub mirror? |
|---|---|---|
| DUSt3R | DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth | YES (naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt) |
| MASt3R | MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth | YES (naver/MASt3R_...catmlpdpt_metric) |
| MUSt3R | MUSt3R_512.pth | **NO** -- direct link only, load_model() takes a PATH |

All three come from download.europe.naverlabs.com/ComputerVision/<MODEL>/.
MUSt3R also ships MUSt3R_512_cvpr.pth (the CVPR checkpoint) -- we default to
MUSt3R_512.pth, the later fine-tune, which its README says outperforms the
CVPR one on most evaluations.

## NEW scripts/fetch_weights.py -- one command

    python scripts/fetch_weights.py --list          # manifest + what's present
    python scripts/fetch_weights.py                 # fetch everything declared
    python scripts/fetch_weights.py --backbones dust3r mast3r must3r
    python scripts/fetch_weights.py --extras        # + optional retrieval files
    python scripts/fetch_weights.py --dir /data/ckpts

It reads `weights_url` / `weights_file` off the adapter classes, so the URL
list can never drift from what the adapters load, and a new backbone becomes
fetchable the moment its adapter declares those two fields.  Resumable
(wget -c), skips what's present, and REJECTS a suspiciously small file --
a truncated download or an HTML error page saved as .pth otherwise surfaces
as an unpickling error hours later.

Files land in third_party/checkpoints/ (override with --dir or
$SMR_CKPT_DIR), which is exactly where the adapters look.

## Resolution order (why local files matter)
  1. explicit model_id= override
  2. local file in $SMR_CKPT_DIR / third_party/checkpoints / checkpoints
  3. HuggingFace hub id, if the model has one
Local wins, so a server holding the .pth never silently re-downloads and the
paper can pin a byte-identical checkpoint.  Both DUSt3R's and MASt3R's
from_pretrained() branch on os.path.isfile(), so handing them a local path
is the documented path, not a hack.

## Two bugs my own tests caught this round
* MUSt3R's weights id was a GUESS in v3.1 ("naver/MUSt3R_512"); there is no
  such hub entry.  Now a local-only .pth with weights_hub=None, and a test
  documents that fact so a future change is noticed.
* Resolution was EAGER in __init__, so merely constructing an adapter blew
  up when the checkpoint was absent.  Now lazy (resolved in _load), because
  metadata listing, preflight and the report script all construct without
  weights.

## Optional retrieval files (declared, not fetched by default)
MASt3R-SfM and MUSt3R's unordered mode use a retrieval model
(trainingfree.pth + codebook.pkl, same directory).  Our pointmap path
doesn't need them; `--extras` fetches them if you want to try that mode.

## RUN ORDER on the server now
    python scripts/fetch_weights.py --backbones dust3r mast3r must3r
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r must3r --max-views 4
The report walks construct -> import -> weights -> load -> infer -> schema
-> bind per backbone and writes JSON; nothing is fatal, so one run tells you
the state of the whole matrix.  Keep --max-views small for DUSt3R: it is
pairwise, so cost is O(K^2) inferences plus a 300-iteration global alignment.

Tests: 50 passed, 6 GPU-tier skipped, no GPU needed.
