# Update v2.1 -- torchvision-compat perceptual loader

Your torchvision predates the VGG16_Weights enum, so --perc-weight was
silently falling back to L1-only (the WARNING + perc 0.0000 columns).
The loader now tries the new enum, then the legacy pretrained=True API.

## Restart the subset run properly (it was only minutes in)
    # in the training screen:
    Ctrl-C
    rm -rf outputs/logs/completion/co3d_subset     # clean slate, same name
    CUDA_VISIBLE_DEVICES=0 python experiments/train_completion.py \
        --data data/completion_co3d --name co3d_subset \
        --val-frac 0.1 --steps 20000 --perc-weight 0.05
    # first eval should show a nonzero perc column and NO warning
    # (first run downloads VGG16 weights once, ~530 MB)

## pi3 harvest in PARALLEL on another GPU (separate screen)
    screen -S harvest_pi3
    cd ~/smr && source .venv/bin/activate
    for cat in ~/co3d_data/*/; do
      CUDA_VISIBLE_DEVICES=1 python scripts/harvest_root.py \
         --root "$cat" --images-sub images --backbone pi3 \
         --out data/completion_co3d \
         --holdouts every:5 --context 1,4,all --max-views 32
    done
Training on GPU 0 and harvest on GPU 1 coexist cleanly; binds are CPU.
