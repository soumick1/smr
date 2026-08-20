# SMR -- Scene Mental Rotation with Emergent Attractor Dynamics

Reference implementation of the SMR project plan: a head-direction-gated
scaffold of continuous attractor fields (3 HD rings + M grid modules), a
Vector-HaSH-style associative memory bound one-shot to frozen-backbone
scene content, and a rotate-by-dynamics readout.  Everything in tiers 1-3
runs on CPU with numpy in minutes; learned backbones (VGGT / Pi3 /
VGGT-Omega) enter only on the GPU server through a thin adapter layer.

## Layout
```
src/smr/
  utils/       geometry (SE(3), ZYX Euler, Euler rates), seeding, reports
  dynamics/    cosine-kernel Amari fields (ring, torus), conjunctive
               velocity, ScaffoldState (place / gate / drive_to / decode)
  memory/      toy Vector-HaSH (T9/T10/T14), field-phase BlockScaffold
               (ridge return, novelty), one-shot RLS binding, store, remap
  backbones/   registry + BackboneOutput; synthetic GT backbone (local),
               vggt / pi3 / vggt_omega adapters (server)
  scene/       synthetic surfel rooms, pinhole z-buffer renderer, surfels
  pipeline/    Algorithm 1 (bind), Algorithm 2 (rotate)
  viz/         house style, figure and GIF writers
experiments/   run_tier1_dynamics.py  run_tier2_memory.py
               run_tier3_integration.py  run_all.py
tests/         fast pytest mirrors of the ladder's core checks
configs/       compact.yaml (local), default.yaml (server)
server/        sync_to_server.sh, setup_backbones.sh
outputs/       figures/  gifs/  reports/   (generated)
```

## Quickstart
```
./setup_env.sh
source .venv/bin/activate
python experiments/run_all.py --fast     # full ladder, ~10-15 min CPU
pytest -q                                # fast unit mirrors
```
Artifacts land in `outputs/`: per-test figures, `ring_integration.gif`,
`torus_bump.gif`, `rotation_sweep.gif` (scaffold activity + recalled
render, live), and JSON reports with every criterion and value.

## The testing ladder (what passing means)
* T1-T8 (dynamics): bump width matches the closed form F1(w/2)=1/J1;
  velocity integration is calibrated and linear below a measured
  omega_max; placement is exact; the HD gate is what buys path invariance
  and kills holonomy; residue decode works and exact-CRT is fragile to
  single-module corruption; reaction time is linear in rotation angle.
* T9-T14 (memory): scaffold size linear in module count; graceful
  pseudoinverse continuum vs the Hopfield cliff; SELECTIVE DETECTABILITY
  (coherent shifts are invisible to the loop, incoherent ones light up the
  reconstruction residual by >10^5) -- the loop is the detector, landmark
  re-anchoring is the corrector, which base-blindness makes mandatory
  anyway; exact recall under cue corruption up to a threshold; novelty
  AUROC 1.0; combinatorial strong generalisation (random 20% of joint
  states locks in the rest; contiguous phase-excluding sets cannot, as the
  negative control shows).
* I1-I5 (integration, synthetic GT backbone): place->decode pose to
  ~0.05 deg / 1% scale; drive_to arrives within tolerance; relocalisation
  from corrupted descriptors; novel-pose splat depth matches ground truth
  to ~0.01 median.

## Backbones on the server
1. `server/sync_to_server.sh` (from WSL, repo root) pushes the tree.
2. On the server: `./setup_env.sh --with-backbones` (installs torch,
   CUDA wheels if `nvidia-smi` is present).
3. `server/setup_backbones.sh` clones the repos into `third_party/`
   (VERIFY the Pi3 / VGGT-Omega URLs before first use).
4. Complete the forward pass in `src/smr/backbones/{vggt,pi3,vggt_omega}.py`
   -- each adapter only has to fill `BackboneOutput`; nothing downstream
   changes (that is the portability claim, exercised here by swapping the
   synthetic backbone for a learned one).

## Wiring VGGT (first learned backbone)
1. `server/setup_backbones.sh`, then `pip install -e third_party/vggt`
   (weights auto-download from the HF hub on first `from_pretrained`).
2. `python scripts/export_synthetic_frames.py` renders the room to PNGs
   plus GT poses/depth at VGGT's native 518 resolution.
3. `python scripts/vggt_sanity.py` runs the adapter on those frames and
   reports Sim(3)-aligned pose and depth errors vs ground truth (the
   plan's week-3 backbone-swap check).  All repo-specific call names in
   `src/smr/backbones/vggt.py` carry VERIFY-ON-SERVER comments -- confirm
   them against your clone once; the adapter fails loudly otherwise.

## Stated assumptions
1. `setup_env.sh` auto-detects CUDA; no driver version is hardcoded.
2. Tiers 1-3 are numpy-only by design so the concept tests run anywhere;
   torch is required only for learned backbones.
3. Backbone repository URLs are verified server-side (see
   `server/setup_backbones.sh`); adapters fail loudly until wired.
