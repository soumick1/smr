# Update v1.8 -- shakedown banked; perceptual loss for the big run

Doc: completion-head shakedown recorded (protocol, metrics incl. the
pass-through invariant, sample figure, L1-blur reading) + the
training-corpus plan paragraph (CO3D + ScanNet++ train; LLFF/7-Scenes
zero-shot).

Code: train_completion.py gains --perc-weight (VGG16 relu3_3 feature
loss on the COMPOSITE -- gradients flow only through hole pixels by
the pass-through rule).  Default 0.0: shakedown runs reproduce exactly.
NOTE: adds a loss_perc column to metrics.csv -- use a fresh --name,
do not --resume an old run with the new script.

## CO3D (start the download in screen; it is large)
    git clone https://github.com/facebookresearch/co3d
    # use their download script; a 10-20 category subset is plenty to
    # start (e.g. hydrant, teddybear, plant, bench, motorcycle, ...)
CO3D layout is category/sequence/images/*.jpg -- point harvest_root at
ONE CATEGORY at a time (it walks one level of scene dirs):
    for cat in hydrant teddybear plant; do \
      python scripts/harvest_root.py --root co3d/$cat --images-sub images \
             --backbone vggt --out data/completion_co3d \
             --holdouts every:5 --context 1,4,all --max-views 32 \
             --limit-scenes 80; done
Then the big training (screen; ~2-3 h at 60k steps):
    python experiments/train_completion.py --data data/completion_co3d \
           --name co3d_v1 --val-frac 0.1 --steps 60000 --perc-weight 0.05


# Fitting co3d on server

screen -S co3ddl
cd ~/smr && python co3d/co3d/download_dataset.py \
    --download_folder ~/co3d_data --single_sequence_subset \
    --clear_archives_after_unpacking --n_download_workers 4 --n_extract_workers 4

Now harvest:

screen -S harvest_co3d
cd ~/smr && source .venv/bin/activate
for cat in ~/co3d_data/*/; do
  python scripts/harvest_root.py --root "$cat" --images-sub images \
     --backbone vggt --out data/completion_co3d \
     --holdouts every:5 --context 1,4,all --max-views 32
done