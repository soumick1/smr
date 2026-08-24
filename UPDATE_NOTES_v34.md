# Update v3.4 -- the last two backbones: scipy alias + faiss import

3/5 became 5/5 (expected).  Both failures were third-party dependency
issues, not adapter bugs, and both are fixed with the narrowest possible
shim rather than a global downgrade that would disturb the other backbones
sharing this environment.

## mast3r: AttributeError: scipy.cluster.hierarchy has no attribute 'distance'

sparse_ga.py does `import scipy.cluster.hierarchy as sch` and then, inside
the default 'hclust-ward' kinematic mode, `sch.distance.squareform(...)`.
Older SciPy re-exported scipy.spatial.distance as that attribute; current
SciPy does not.  Restoring the alias is precisely the old behaviour --
`sch.distance.squareform` IS `scipy.spatial.distance.squareform`.

  * `patch_scipy_hierarchy_distance()` applies it, idempotently, and is a
    no-op where SciPy still has the alias (this container does, your server
    does not -- hence the shim rather than a version pin).
  * `kinematic_mode` is now an adapter argument.  If the shim ever fails,
    `get_backbone("mast3r", kinematic_mode="mst")` is upstream's own
    scipy-free alternative, reachable without editing code.

## must3r: ModuleNotFoundError: No module named 'faiss'

must3r/demo/inference.py imports the retrieval `Retriever` at MODULE level,
and must3r/retrieval/processor.py runs `import faiss;
faiss.StandardGpuResources()` at import time -- so importing the inference
module needs faiss even though we pass retrieval=None and run sequence mode,
which never retrieves.

Their README's real recipe is `pip install faiss-cpu` PLUS building asmk
from source (git clone jenicek/asmk, cythonize, pip install).  Note faiss-cpu
alone is not sufficient for their code path: it lacks StandardGpuResources,
so their except-branch then requires asmk.

Since we never use retrieval, the adapter installs a minimal `faiss` stub
that satisfies the import, prints a one-line note that it did so, and NEVER
shadows a real installation.  If you want MUSt3R's unordered/retrieval mode
later, follow their README steps and the stub silently stops being used.

## Re-run
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r must3r --max-views 4

If must3r still fails, it will now be at the point I flagged as genuinely
unpinnable from the repo -- the shape of must3r_inference's return value --
and the error will describe what it actually found so I can finish it in one
pass.

Tests: 63 passed, 6 GPU-tier skipped (4 new: the scipy shim under simulated
server conditions, stub-never-shadows-real, stub-satisfies-import, and the
scipy-free fallback mode staying reachable).
