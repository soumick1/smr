# smr_updates186 — bank growth, retrieval capacity and pruning (reviewer question, continuous mapping)

`scripts/bank_scaling.py` answers the question with the production memory code and no backbone:
* one continuous-mapping bank from ALL cached 7-Scenes sequences (seq-01 runs + 18-trajectory split = 25 sequences of
  7 rooms; sequences of a room share its world frame, so later sequences revisit earlier ones; rooms placed on a grid in
  scaffold units; GT poses at 1.5 m per unit); the last sequence of every room is held out as queries;
* retrieval recall@1 / @5 against GT revisits (strict 10 % diag + 30 deg; covis 30 % + 60 deg) after every bound
  sequence, scaffold index vs flat cosine index, wrong-room top-1 rate, query latency;
* bytes: constant RLS maps (47 MB at N_h 2048, D 384), per-entry index record (19.7 kB as implemented; 3.3 kB with the
  address packed as 128 int16 indices), dense content per entry if kept (3.09 MB at 518x392, 80 % points, 19 B/point);
* pruning at bind time, each on the full bank: pose spacing 0.10 / 0.25 / 0.50 m (+15 deg), scaffold-cell deduplication
  (address overlap >= 0.5 / 0.7 and cue cosine >= 0.8), cue deduplication (cosine >= 0.95 / 0.90): entries kept, recall,
  index MB, dense GB.

```bash
unzip -o smr_updates186.zip
ls outputs/cache | grep desc_dino | head -3                  # the descriptor caches the script pairs with data/gt/*.npz
PYTHONPATH=src python scripts/bank_scaling.py --gt-dir data/gt --cache-dir outputs/cache --dataset 7scenes --md outputs/bank_scaling.md --json outputs/bank_scaling.json
```
It prints the (GT, descriptor) pairs it found first; if that list is short or empty, send `ls outputs/cache` and I will adjust
the pairing. Send back `outputs/bank_scaling.md`.

Expected (P92-P94): scaffold recall@1 falls as the bank passes a few hundred entries (the cue-to-address map is a
384-dimensional least-squares regression) while the flat index holds; wrong-room top-1 stays below 5 % with the rooms
8 units apart; scaffold-cell deduplication removes >= 60 % of entries at <= 0.02 recall@5 cost, pose spacing 0.25 m
removes ~70 % at a larger cost.
