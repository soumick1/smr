#!/usr/bin/env bash
# Clone + install the frozen geometry backbones into third_party/.
#
#   bash server/setup_backbones.sh                 # shows this list
#   bash server/setup_backbones.sh group1          # dust3r mast3r
#   bash server/setup_backbones.sh group2          # fast3r stream3r cut3r mvdust3r
#   bash server/setup_backbones.sh fast3r stream3r # any subset
#   bash server/setup_backbones.sh all
#
# INSTALL POLICY: packages are installed with --no-deps on purpose.
# fast3r's requirements.txt pins numpy<2.0.0; installing it wholesale would
# downgrade the numpy that VGGT, pi3, DUSt3R, MASt3R and our own code are
# using.  We add only what the inference path actually imports.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party && cd third_party
TP="$(pwd)"

clone () { if [ -d "$2" ]; then echo "[$2] already present";
           else git clone --recursive "$1" "$2" || echo "[$2] CLONE FAILED"; fi }
inst  () { pip install -e "$1" --no-deps 2>/dev/null \
           && echo "[$(basename "$1")] installed (--no-deps)" \
           || echo "[$(basename "$1")] no setup.py; PYTHONPATH-style (the "\
"adapter adds it to sys.path automatically)"; }

want="${*:-}"
if [ -z "$want" ]; then
  cat <<'USAGE'
usage: bash server/setup_backbones.sh <targets>
  group1   dust3r mast3r                (PYTHONPATH-style, no setup.py)
  group2   fast3r stream3r cut3r mvdust3r
  all      everything
  or name any subset, e.g.:  fast3r stream3r
USAGE
  exit 0
fi
[[ "$want" == "group1" ]] && want="dust3r mast3r"
[[ "$want" == "group2" ]] && want="fast3r stream3r cut3r mvdust3r"
has () { [[ " $want " == *" $1 "* || " $want " == *" all "* ]]; }

# ---- already integrated -----------------------------------------------
if has vggt   ; then clone https://github.com/facebookresearch/vggt.git vggt; fi
if has pi3    ; then clone https://github.com/yyfz/Pi3.git Pi3; inst "$TP/Pi3"; fi

# ---- group 1: DUSt3R family (no setup.py -> sys.path only) ------------
if has dust3r ; then clone https://github.com/naver/dust3r.git dust3r; fi
if has mast3r ; then clone https://github.com/naver/mast3r.git mast3r; fi

# ---- group 2 ----------------------------------------------------------
# fast3r + stream3r are Lightning projects: the inference path needs
# `lightning` (and einops), but NOT the full requirements file.
if has fast3r || has stream3r; then
  # --no-deps IS NOT OPTIONAL HERE.  Installing lightning normally pulls a
  # newer torch (CUDA-13 build) which this driver cannot run, and fast3r's
  # requirements.txt additionally pins numpy<2.  Both break every other
  # backbone.  Install bare, then add missing pure-python packages one at a
  # time as the checker reports them.
  echo "[deps] installing WITHOUT dependencies to protect torch/numpy"
  pip install --no-deps "lightning>=2.5.0" lightning-utilities lightning-bolts \
      torchmetrics einops omegaconf hydra-core rootutils \
      antlr4-python3-runtime packaging typing-extensions fsspec pyyaml \
      2>/dev/null && echo "[deps] ok" || echo "[deps] see notes"
  python - <<'PYCHK'
import torch
print(f"[check] torch {torch.__version__}, cuda build {torch.version.cuda}, "
      f"available={torch.cuda.is_available()}")
assert torch.cuda.is_available(), (
    "TORCH BROKE: run  bash server/restore_env.sh --fix")
PYCHK
fi
if has fast3r   ; then clone https://github.com/facebookresearch/fast3r.git fast3r
                       inst "$TP/fast3r"; fi
if has stream3r ; then clone https://github.com/NIRVANALAN/STream3R.git STream3R
                       inst "$TP/STream3R"; fi
if has cut3r    ; then clone https://github.com/CUT3R/CUT3R.git CUT3R
                       echo "[cut3r] weights are Google Drive only:"
                       echo "  pip install gdown && cd $TP/checkpoints && \\"
                       echo "  gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view"; fi
if has mvdust3r ; then clone https://github.com/facebookresearch/mvdust3r.git mvdust3r; fi

echo
echo "cloned into $TP"
echo "next:"
echo "  python scripts/fetch_weights.py --list"
echo "  python scripts/check_backbones.py --frames lab_photos/room/images_8 \\"
echo "      --backbones vggt pi3 dust3r mast3r fast3r stream3r --max-views 4"
