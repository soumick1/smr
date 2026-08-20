#!/usr/bin/env bash
# Push this repo to the GPU server (run from WSL, inside the repo root).
# Excludes the local venv and generated outputs; re-run any time.
set -euo pipefail
rsync -avz \
  --exclude '.venv' --exclude 'outputs' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude '*.egg-info' \
  -e "ssh -p 22" \
  ./ soumick@10.97.144.63:~/smr/
echo "Synced.  On the server:  cd ~/smr && ./setup_env.sh --with-backbones"
