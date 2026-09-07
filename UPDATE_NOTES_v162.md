# smr_updates162 — Table 3 with Objaverse-LVIS (held-out) AND GSO columns

Table 3 now has two column groups per row: **Objaverse-LVIS held out** (the 1 % of training objects the head never
saw; our renders; LVSM's object protocol) and **GSO** (1,033 objects). GSO cells are final; the Objaverse cells need
one evaluation pass (raw and read, four backbones, ~35 min on one GPU) -- the training logs only have raw validation.
```bash
cd ~/smr && unzip -o smr_updates162.zip && source .venv/bin/activate
DATA=~/data/nvs/objaverse_c70 CUDA_VISIBLE_DEVICES=0 bash scripts/nvs/eval_val_all.sh     # -> outputs/nvs/summary_val.txt
python scripts/nvs/fill_tab_nvs.py                                                       # fills paper/tab_nvs.tex, bolds the better of each pair
cat outputs/nvs/summary_val.txt
```
Send `summary_val.txt` (or the filled tab_nvs.tex) and I write the one sentence Sec. 5.4 needs about the held-out
column and return the Overleaf zip. Expect the held-out numbers to be ~0.5-1 dB above GSO (in-distribution renders)
with the same read pattern: VGGT/VGGT-Omega up, pi3 identical, STream3R flat or down.
`--val-only` uses exactly the trainer's split rule (crc32(name) % 10000 < 100); the evaluator's 4 inputs / 10 targets
are the first 14 of the 16 training views of each held-out object (same views for raw and read).
