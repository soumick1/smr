# smr_updates204 — figures with s.e.m. bars and no PGO; CO3Dv2 as the test-only NVS set

Files (unpack at the repo root): `scripts/fig_nvs_seq.py`, `scripts/nvs_seq_co3d.sh`, `experiments/nvs_seq_cache.py`.
* Fig. S3(b): mean ± standard error of the per-target gain (the 95 % intervals were ±0.3 dB around +0.5 dB because
  per-target PSNR varies by ~2 dB; s.e.m. over 156 targets is ~0.16 dB). Fig. S3(c) adds the same for CO3Dv2 once its
  results exist; the per-sequence scatter moves to (d).
* Fig. S2 (both variants) and Fig. 2B: ground truth / raw / +SMoRe only (no PGO column).
* CO3Dv2 test-only protocol: the 37 validated orbits, 200 frames at stride 1, every 8th frame held out; the decoders
  trained on 7-Scenes are applied unchanged (zero-shot, as Objaverse -> GSO before). Portrait orbits keep their
  orientation (288 x 384), intrinsics come from the npz when stored, otherwise from the backbone.

```bash
cd ~/smr && unzip -o ~/smr_updates204.zip && source .venv/bin/activate
screen -dmS co3dnvs bash -c 'cd ~/smr && bash scripts/nvs_seq_co3d.sh 2>&1 | tee outputs/nvsq_co3d.log; exec bash'   # ~3 h, 3 GPUs
# when it prints ALL DONE (and after the step-1/2 of v203 for the 7-Scenes renders):
python scripts/fig_nvs_seq.py --root outputs/nvs_seq_v2 --root-co3d outputs/nvs_seq_v2_co3d --caches cache/nvsseq --out outputs/figures/nvs_seq \
    --old-fig2 images/Fig3_mod.png --bb vggt_omega --fig2-seq office_seq01 --fig2-target 128
```
Send `outputs/nvsq_co3d.log` (six tables at the end) and the figures folder. The table for the paper then has two blocks,
7-Scenes (decoders trained on the other sequences of the same scenes) and CO3Dv2 (test only), raw vs +SMoRe, mean ± s.e.m.
