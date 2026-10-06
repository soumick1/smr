# smr_updates198 — core aligned with the manuscript (v25); repeated reads removed

Full files from the pushed HEAD (ed792d0) with the changes below; unpack at the repo root, review with `git diff`.

## Default behaviour now follows the manuscript (`--paper`, the default; `--legacy` reproduces every earlier report)
| manuscript statement | implementation |
|---|---|
| scaffold state: 3 modules (48x48 torus + 48 height ring) + 3 orientation rings of 256 units, N_g = 7,824 | `BlockScaffold(ring_N=256)`, yaw/pitch/roll (ZYX) encoded as wrapped-Gaussian ring templates |
| address: W_gh in R^{1024x7824} ~ N(0, 1/7824), 64 largest kept, L2-normalised | `--N-h 1024 --k 64` (W_gh was already N(0,1/N_g); top-k and normalisation unchanged) |
| positions in scene units with median scene depth = 1 | `--scene-unit auto`: median predicted depth of the first window from one backbone pass; `pos_scale` in the index |
| retrieved pair = frame +-3 from the same source window | `pair_strict`: exactly +-3 (the legacy fallback to +-2 / +-1 is off) |
| two pairs from different source windows, else no correction | `require_two_sites` (single-pair closures and the tight budget are no longer used) |
| accept iff the two proposals agree: rotation <= 3 deg, position <= 0.15 E_k | `--site-agree 3,0.15` with `reject_on_disagree` (disagreement rejects the revisit; no demotion). Per-pair validity gates and the drift budget are OFF in this preset; `--paper-gates` keeps them |
| T_k^corr: joint robust fit of the four retrieved frames | unchanged |
| progressive Sim(3) interpolation between the reference and the current window | `--correction distribute` (Eq. online); the legacy local pose-graph relaxation is `--correction relax` |
| App. A: five reweighting rounds with w = min(1, 2.5 median(r)/r), orientation rejection, scale gate gamma_s = 1.5 on consecutive windows | `sim3.fit_poses_irls` (rotation still from the frames' orientations by weighted Procrustes, since two centres do not determine it), `seq_scale_gate=1.5` with fallback to the previous window's scale |
| joint-pass ("anchored") local geometry is the main policy | unchanged (`--local-from anchored`) |

## Removed
Repeated reads (consensus over input orderings): `smr.nvs.geometry.predict_geometry` is single-pass and refuses `reads > 1`;
`eval_gso.py` and `points_suite.py` refuse `--reads > 1`. Dense geometry now comes only from the stored windows and their
revised placements (the sequence NVS / dense-map code that follows this package).

## Tests
`tests/test_guarantees.py`: 13 pass (new: paper acceptance rules; IRLS fit; the defaults test now checks the preset).
On the heavy-noise synthetic world the paper preset accepts no closure (the two proposals never agree within 3 deg);
that is a property of that toy, the real-data effect is what the sweep below measures.

## Run (what the alignment costs; ~1 h on one GPU, new addresses mean some new anchored passes)
```bash
cd ~/smr && unzip -o ~/smr_updates198.zip && git status --short
PYTHONPATH=src python -m pytest -q tests/test_guarantees.py          # 13 pass
CUDA_VISIBLE_DEVICES=0 bash scripts/paper_config_sweep.sh
python scripts/gate_table.py --root outputs/ablate/paper_config --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --md outputs/paper_config.md
for v in paper paper_gates paper_posonly paper_relax paper_oldaddr; do echo == $v; python scripts/paired_stats.py reports --a 'outputs/ablate/paper_config/7scenes/legacy/*.json' --b "outputs/ablate/paper_config/7scenes/$v/*.json" --row smr --metric ate_rmse | tail -2; python scripts/paired_stats.py reports --a 'outputs/ablate/paper_config/co3d/legacy/*.json' --b "outputs/ablate/paper_config/co3d/$v/*.json" --row smr --metric auc30 | tail -2; done
```
Send `outputs/paper_config.md` and the loop output. The `legacy` arm must reproduce 0.030 / 0.029 and 93.4 / 93.3; the
`paper` arm is the configuration the manuscript describes, and the three variants say which change carries any difference.
If the agreement test alone lets false closures through (paper vs paper_gates), the manuscript gets one sentence on pair
validity; if the orientation rings hurt retrieval (paper vs paper_posonly), the manuscript's scaffold state should be
described as position-addressed. The main tables are regenerated only after this decision.
