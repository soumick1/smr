#!/usr/bin/env python3
"""Fetch backbone checkpoints, driven by the adapter registry.

Every URL lives in exactly one place -- the `weights_url` / `weights_file`
attributes on each adapter class -- so this script cannot drift from what
the adapters actually load.  Adding a backbone automatically adds it here.

    python scripts/fetch_weights.py --list
    python scripts/fetch_weights.py                       # every backbone
    python scripts/fetch_weights.py --backbones dust3r mast3r must3r
    python scripts/fetch_weights.py --extras               # + retrieval files
    python scripts/fetch_weights.py --dir /data/checkpoints
    python scripts/fetch_weights.py --dry-run

Downloads land in the first of: --dir, $SMR_CKPT_DIR,
third_party/checkpoints -- which is exactly where the adapters look, and
local files always win over the HuggingFace hub so a machine that has the
.pth never silently re-downloads and the paper can pin one checkpoint.

Resumable (wget -c, or urllib with a Range request), skip-if-present, and
it refuses to accept a suspiciously small file: a truncated download or an
HTML error page saved as a .pth is a real failure mode that otherwise
surfaces as an unpickling error hours later.
"""
import argparse, os, pathlib, re, shutil, subprocess, sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smr.backbones.base import _REGISTRY          # noqa: E402
from smr.backbones.pointmap import checkpoint_dirs  # noqa: E402

MIN_BYTES = 10 * 1024 * 1024        # a real ViT-L checkpoint is >> 10 MB


def entries(names=None, extras=False):
    """(backbone, filename, url, required) from the registry itself.

    `url` may be an https link or a Google Drive page; download() picks the
    right transport.  Two repos (CUT3R, MonST3R) publish only via Drive --
    that is their own documented recipe, not a workaround.
    """
    out = []
    for name in sorted(_REGISTRY):
        if names and name not in names:
            continue
        cls = _REGISTRY[name]
        url = getattr(cls, "weights_url", None) or getattr(
            cls, "weights_gdrive", None)
        fn = getattr(cls, "weights_file", None)
        if url and fn:
            out.append((name, fn, url, True))
        if extras:
            for fn_e, url_e in getattr(cls, "weights_extras", ()) or ():
                out.append((name, fn_e, url_e, False))
    return out


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def download(url, dest, dry_run=False):
    if dry_run:
        return "would download"
    tmp = dest.with_suffix(dest.suffix + ".part")
    if "drive.google.com" in url:
        if not shutil.which("gdown"):
            raise RuntimeError("needs gdown for Google Drive: "
                               "pip install gdown")
        # Pass the bare FILE ID, not the share URL: gdown 6.x removed
        # --fuzzy (URL parsing became the default) while older versions
        # REQUIRE it, so a plain id is the only spelling every version
        # accepts.
        m = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
        gid = m.group(1) if m else url
        rc = subprocess.run(["gdown", gid, "-O", str(tmp)]).returncode
        if rc != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"gdown exited {rc}")
        size = tmp.stat().st_size
        if size < MIN_BYTES:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"gdown produced only {human(size)}")
        tmp.rename(dest)
        return f"ok {human(size)}"
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    hdr = (["--header", f"Authorization: Bearer {token}"]
           if token and "huggingface.co" in url else [])
    if shutil.which("wget"):
        cmd = ["wget", "-c", "-q", "--show-progress", *hdr, url, "-O",
               str(tmp)]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"wget exited {rc}")
    else:                                       # stdlib fallback, resumable
        start = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if token and "huggingface.co" in url:
            req.add_header("Authorization", f"Bearer {token}")
        if start:
            req.add_header("Range", f"bytes={start}-")
        with urllib.request.urlopen(req) as r, open(tmp, "ab") as f:
            shutil.copyfileobj(r, f)
    size = tmp.stat().st_size
    if size < MIN_BYTES:
        head = tmp.read_bytes()[:200]
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded only {human(size)} -- almost certainly not a "
            f"checkpoint. First bytes: {head[:120]!r}")
    tmp.rename(dest)
    return f"ok {human(size)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--extras", action="store_true",
                    help="also fetch optional retrieval files")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = entries(a.backbones, a.extras)
    if not rows:
        print("nothing to fetch (no adapter declares weights_url for "
              f"{a.backbones or 'any backbone'})")
        return 0

    dest_dir = pathlib.Path(a.dir) if a.dir else checkpoint_dirs()[0]
    if a.list:
        print(f"target directory: {dest_dir}\n")
        for name, fn, url, req in rows:
            here = (dest_dir / fn).exists()
            print(f"  [{'x' if here else ' '}] {name:<10} "
                  f"{'' if req else '(optional) '}{fn}\n      {url}")
        # backbones with no automatable URL still need to be discoverable
        for name in sorted(_REGISTRY):
            if a.backbones and name not in a.backbones:
                continue
            cls = _REGISTRY[name]
            if getattr(cls, "weights_url", None):
                continue
            hub = getattr(cls, "weights_hub", None)
            note = getattr(cls, "weights_note", None)
            if hub:
                print(f"  [auto] {name:<10} downloaded on first use from the "
                      f"HuggingFace hub: {hub}")
            elif note:
                print(f"  [MANUAL] {name:<10} {note}")
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"target directory: {dest_dir}")
    print("local files win over the HuggingFace hub, so this makes the "
          "checkpoint reproducible.\n")

    results = []
    for name, fn, url, req in rows:
        dest = dest_dir / fn
        if dest.exists() and dest.stat().st_size >= MIN_BYTES:
            results.append((name, fn, f"present {human(dest.stat().st_size)}"))
            print(f"[skip] {name}: {fn} already present")
            continue
        print(f"[get ] {name}: {fn}")
        try:
            results.append((name, fn, download(url, dest, a.dry_run)))
        except Exception as e:
            results.append((name, fn, f"FAILED: {type(e).__name__}: {e}"))
            print(f"  !! {results[-1][2]}")

    print("\n" + "=" * 74)
    for name, fn, status in results:
        short = fn if len(fn) <= 46 else fn[:21] + "..." + fn[-22:]
        print(f"{name:<11} {short:<47} {status}")
    bad = [r for r in results if r[2].startswith("FAILED")]
    print("=" * 74)
    print(f"{len(results) - len(bad)}/{len(results)} ready"
          + (f"; {len(bad)} failed" if bad else ""))
    print("\nnext: python scripts/check_backbones.py "
          "--frames lab_photos/room/images_8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
