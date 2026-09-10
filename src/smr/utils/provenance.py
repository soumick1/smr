"""Run provenance for every report (v177, plan Task 1).

    from smr.utils import provenance
    report["provenance"] = provenance.run_record(args, backbone="vggt_omega")

Records: git commit / branch / dirty state of the repository that produced the
run, the SHA-256 (first 16 hex) + size + mtime of the checkpoint files
involved (all *.pt/*.pth under third_party/checkpoints and checkpoints/, or
only the one named by `weights_file`), hostname, Python / numpy / torch
versions, UTC timestamp, argv and the parsed arguments.  Hashing a
multi-GB checkpoint takes seconds, so hashes are cached in
~/.cache/smr_provenance keyed by (path, size, mtime) and computed once.
Never raises: a missing tool yields a "unknown" field, never a failed run.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import pathlib
import platform
import socket
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CACHE = pathlib.Path(os.environ.get("SMR_PROVENANCE_CACHE", pathlib.Path.home() / ".cache" / "smr_provenance"))
CKPT_DIRS = (REPO_ROOT / "third_party" / "checkpoints", REPO_ROOT / "checkpoints")


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
                              check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def git_state():
    commit = _git("rev-parse", "HEAD") or "unknown"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    status = _git("status", "--porcelain", "--untracked-files=no")
    return dict(commit=commit, branch=branch, dirty=bool(status),
                dirty_files=[l[3:] for l in status.splitlines()][:40] if status else [])


def file_hash(path, prefix_hex=16):
    """SHA-256 of a file, cached by (path, size, mtime)."""
    p = pathlib.Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    key = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{key}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text())
        except Exception:  # noqa: BLE001
            pass
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    rec = dict(path=str(p), sha256=h.hexdigest()[:prefix_hex], size=st.st_size,
               mtime=_dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).isoformat(timespec="seconds"))
    cf.write_text(json.dumps(rec))
    return rec


def checkpoint_records(weights_file=None, dirs=CKPT_DIRS):
    out = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix not in (".pt", ".pth", ".safetensors", ".ckpt", ".bin"):
                continue
            if weights_file and p.name != weights_file:
                continue
            r = file_hash(p)
            if r:
                out.append(r)
    return out


def env_state():
    try:
        import numpy as np
        npv = np.__version__
    except Exception:  # noqa: BLE001
        npv = "unknown"
    try:
        import torch
        tv = torch.__version__
        cuda = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        tv, cuda, gpu = "not importable", None, None
    return dict(host=socket.gethostname(), python=platform.python_version(), numpy=npv, torch=tv, cuda=cuda, gpu=gpu,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def run_record(args=None, backbone=None, weights_file=None, hash_checkpoints=True):
    """The dict to store under report["provenance"].  `args` may be an
    argparse.Namespace or a dict; non-JSON values are stringified."""
    a = vars(args) if hasattr(args, "__dict__") else dict(args or {})
    a = {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)) for k, v in a.items()}
    if weights_file is None and backbone:
        try:
            from smr.backbones import get_backbone_class  # optional helper; fall back to all files
            weights_file = getattr(get_backbone_class(backbone), "weights_file", None)
        except Exception:  # noqa: BLE001
            weights_file = None
    return dict(version=1, timestamp_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                git=git_state(), env=env_state(), argv=sys.argv, args=a, backbone=backbone,
                checkpoints=(checkpoint_records(weights_file) if hash_checkpoints else []),
                weights_file=weights_file)


if __name__ == "__main__":
    print(json.dumps(run_record({"demo": True}), indent=1))
