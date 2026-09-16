# smr_updates183 — two small fixes before the VPR sweep

1. `torch.hub.load(..., trust_repo=True)`: the hub asked "Do you trust this repository (y/N)?" during your smoke test and you
   answered by hand; inside `vpr_sweep.sh` (stdout to a log, no tty) the prompt would raise and the SALAD and BoQ arms would be
   skipped after two failures. EigenPlaces and CosPlace are already on your trusted list; the others no longer need it.
2. The smoke test takes frames from a GT npz (`image_paths`) instead of a guessed path, and tests several kinds in one call.

Everything else from your log is fine: EigenPlaces (2048-D) and CosPlace weights downloaded, hloc installed (NetVLAD weights
download on first use), MixVPR repo cloned with pytorch_lightning present. MixVPR still needs the authors' checkpoint:
download `resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt` from the link in third_party/MixVPR/README.md (Google Drive,
`gdown <id>` works) into third_party/checkpoints/; without it the `mixvpr` arm is skipped, which is acceptable.

## Run
```bash
unzip -o smr_updates183.zip
PYTHONPATH=src python -m smr.stitch.vpr_descriptors dino eigenplaces cosplace salad@4096 boq@4096 netvlad mixvpr dino+eigenplaces --gt data/gt/7scenes_chess_seq01.npz
#   expect one line per kind: dim, cos(0,20) > cos(0,500), "ok"; FAILED lines name the missing install
CUDA_VISIBLE_DEVICES=0 bash scripts/vpr_sweep.sh
for m in anchored plain; do python scripts/gate_table.py --root outputs/ablate/vpr/$m --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --md outputs/vpr_$m.md; done
for arm in eigenplaces cosplace salad@4096 dino+eigenplaces; do echo == $arm; python scripts/paired_stats.py reports --a 'outputs/ablate/vpr/plain/7scenes/dino/*.json' --b "outputs/ablate/vpr/plain/7scenes/$arm/*.json" --row smr --metric ate_rmse --per-item | tail -9; done
for v in plain_seed0 plain_seed1; do python scripts/closure_reliability.py --glob "outputs/ablate/localfrom/7scenes/$v/*.json" --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --est-dir outputs/ablate/localfrom/7scenes/$v --md outputs/rel_$v.md | tail -45; done
```
If an arm still fails in the sweep, its `.log` under outputs/ablate/vpr/<mode>/<dataset>/<arm>/ has the reason.
