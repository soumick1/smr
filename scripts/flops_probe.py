#!/usr/bin/env python3
"""Measure compute resources per (backbone, N) on ONE fixed machine.

For each backbone and each N in --n-list, runs ONE forward pass on the
first N frames of --gt (a real npz) and records:
  gflops         forward FLOPs / 1e9 (torch FlopCounterMode)
  gpu_peak_gb    torch.cuda.max_memory_allocated after the pass
  ram_peak_gb    process peak RSS (ru_maxrss) delta
  wall_s         wall-clock seconds for the pass
  cpu_s          process CPU seconds (time.process_time) for the pass
  status         ok | OOM | error:<msg>
Also probes the windowed unit costs used by +SMR rows: N=32 (one window)
and N=36 (anchored window: 32 + 2 sites x 2 frames), so composed costs
can be reported as  n_windows x c32 (+ n_closures x c36).

    python scripts/flops_probe.py --gt data/gt/co3d_full/apple_110*.npz \
        --backbones vggt,vggt_omega,pi3,fast3r,dust3r,mast3r,streamvggt,stream3r \
        --n-list 10,32,36,50,100,200 --json outputs/reports/compute_probe.json

FLOP counting notes: FlopCounterMode covers matmul/conv/attention ops;
custom CUDA kernels outside torch ops are not counted (states 'partial'
in the report if the counter saw zero ops). One warm-up pass at N=2 is
run first so lazy weight loading does not pollute timings.
"""
import argparse, glob, json, resource, sys, time
import numpy as np

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kB->GB (linux)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="npz with image_paths (>=max N frames)")
    ap.add_argument("--backbones", required=True)
    ap.add_argument("--n-list", default="10,32,36,50,100,200")
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    from torch.utils.flop_counter import FlopCounterMode
    sys.path.insert(0, "src")
    from smr.backbones import get_backbone

    d = np.load(glob.glob(a.gt)[0], allow_pickle=True)
    paths = [str(p) for p in d["image_paths"]]
    ns = [int(x) for x in a.n_list.split(",")]
    assert len(paths) >= max(ns), f"npz has {len(paths)} frames < max N {max(ns)}"

    out = {"gt": a.gt, "device": torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
           "rows": []}
    for name in a.backbones.split(","):
        name = name.strip()
        try:
            bb = get_backbone(name)
        except Exception as e:                                   # noqa: BLE001
            out["rows"].append(dict(backbone=name, status=f"error:load:{e}")); continue
        try:
            bb.infer(paths[:2])                                  # warm-up / weight load
        except Exception as e:                                   # noqa: BLE001
            out["rows"].append(dict(backbone=name, status=f"error:warmup:{e}")); continue
        for N in ns:
            row = dict(backbone=name, N=N)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            r0, t0, c0 = rss_gb(), time.time(), time.process_time()
            try:
                bb.infer(paths[:N])                      # measured pass: time/mem only
                row.update(status="ok",
                           wall_s=round(time.time() - t0, 2),
                           cpu_s=round(time.process_time() - c0, 2),
                           gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           ram_peak_gb=round(max(0.0, rss_gb() - r0), 2))
                # FLOPs on a separate counted pass; inference_mode breaks the
                # counter in some torch builds, so alias it to no_grad here.
                try:
                    import torch.utils.module_tracker as _mt
                    class _NoTrack:                      # totals only; no hooks
                        parents = {"Global"}
                        def __enter__(self): return self
                        def __exit__(self, *a): return False
                    import torch.utils.flop_counter as _fc
                    _MT = _mt.ModuleTracker
                    _mt.ModuleTracker = _NoTrack
                    _FCMT = getattr(_fc, "ModuleTracker", None)
                    if _FCMT is not None:
                        _fc.ModuleTracker = _NoTrack        # from-import binding
                    try:
                        fl_mode = FlopCounterMode(display=False)
                        if hasattr(fl_mode, "mod_tracker"):  # belt and braces:
                            fl_mode.mod_tracker = _NoTrack() # replace on instance
                        with fl_mode:
                            bb.infer(paths[:N])
                        fl = int(fl_mode.get_total_flops())
                        row.update(gflops=round(fl / 1e9, 1),
                                   flops_partial=bool(fl == 0))
                    finally:
                        _mt.ModuleTracker = _MT
                        if _FCMT is not None:
                            _fc.ModuleTracker = _FCMT
                except Exception as e:                   # noqa: BLE001
                    row.update(gflops=None, flops_note=f"counter:{type(e).__name__}")
            except torch.cuda.OutOfMemoryError as e:
                row.update(status="OOM", gpu_req=str(e).split("Tried to allocate")[-1][:20],
                           wall_s=round(time.time() - t0, 2))
                torch.cuda.empty_cache()
            except Exception as e:                       # noqa: BLE001
                row.update(status=f"error:{type(e).__name__}:{str(e)[:80]}")
                torch.cuda.empty_cache()
            out["rows"].append(row)
            def _f(k, w):
                v = row.get(k)
                return f"{v:>{w}}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
            print(f"{name:<12} N={N:<4} {row['status']:<10} "
                  f"{_f('gflops',10)} GF  {_f('gpu_peak_gb',6)} GB  {_f('wall_s',7)} s"
                  + (f"  [{row['flops_note']}]" if row.get('flops_note') else ""))
        del bb; torch.cuda.empty_cache()
    json.dump(out, open(a.json, "w"), indent=1)
    print("->", a.json)

if __name__ == "__main__":
    main()
