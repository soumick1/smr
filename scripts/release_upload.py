#!/usr/bin/env python3
"""Upload the release artifacts to Hugging Face (models + results -> ZhangLab-DeepNeuroCogLab/SMoRe; GT files -> ZhangLab-DeepNeuroCogLab/SMoRe-data).

    pip install huggingface_hub && huggingface-cli login
    python scripts/release_upload.py --dry-run
    python scripts/release_upload.py [--model-repo ZhangLab-DeepNeuroCogLab/SMoRe] [--data-repo ZhangLab-DeepNeuroCogLab/SMoRe-data]

Run from the repo root on the machine that holds ckpt/, outputs/ and data/gt/. Uploads are resumable; re-running skips
files already present. Writes a README (model card) into each repo if none exists.
"""
import argparse, glob, os, pathlib

MODEL_FILES = [("ckpt/nvsseq", "*.pt", "decoders"), ("outputs/reports", "t1axis_*_s*.json", "results/pose_length"),
               ("outputs/reports", "t1b_*.json", "results/pose_n200"), ("outputs/reports", "suite_pose_re10k*.json", "results/pose_re10k_n10"),
               ("outputs/nvs_seq_v2", "*/*.jsonl", "results/nvs_7scenes"), ("outputs/nvs_seq_v2_co3d", "*/*.jsonl", "results/nvs_co3d"),
               ("outputs/nvs_seq_v2", "*/eval.log", "results/nvs_7scenes"), ("outputs/nvs_seq_v2_co3d", "*/eval.log", "results/nvs_co3d"),
               ("outputs2/reports", "pilotA_7scenes_*_seq01_*_s5_c32.json", "results/slam"), ("outputs/reports", "compute_probe.json", "results/compute")]
DATA_FILES = [("data/gt", "7scenes_*.npz", "gt/7scenes"), ("data/gt/co3d_full", "*.npz", "gt/co3d_full"), ("data/gt", "re10k*/*.npz", "gt/re10k"),
              ("data/gt", "tum_*.npz", "gt/tum"), ("data", "re10k/*.txt", "lists/re10k"), ("data", "dtu/*.txt", "lists/dtu")]
MODEL_CARD = """---
license: mit
tags: [3d-reconstruction, camera-pose, novel-view-synthesis, memory]
---
# SMoRe: Scaffold Memory for Revision
Artifacts of *Remember, Revisit, Revise: Scaffold Memory for Long-Horizon 3D Geometry* (Sarker and Zhang, 2026).
* `decoders/`: sequence Gaussian decoders for novel view synthesis, two per geometry model (`<model>_raw.pt` trained on
  chained geometry, `<model>_smr.pt` on SMoRe geometry); load with `experiments/nvs_seq_decoder.py eval`.
* `results/`: per-sequence report files behind the paper's tables, so every table regenerates without a GPU
  (`scripts/pilot_a_table.py`, `scripts/nvs_table.py`, `scripts/sevenscenes_table.py`).
Code: https://github.com/ZhangLab-DeepNeuroCogLab/SMoRe
"""
DATA_CARD = """---
license: other
tags: [3d-reconstruction, camera-pose, evaluation-subsets]
---
# SMoRe evaluation subsets
Ground-truth pose files and frame lists (`*.npz`: `poses` (N,4,4) camera-to-world, `image_paths` relative to the dataset
root) for the validated subsets used in the paper: 7-Scenes (all sequences), 37 CO3Dv2 trajectories, RE10K clip lists,
TUM. Images are not redistributed; obtain each dataset from its source and point the paths at it
(`scripts/prepare_standard_sets.sh`). Code: https://github.com/ZhangLab-DeepNeuroCogLab/SMoRe
"""


def collect(spec):
    out = []
    for base, pat, dest in spec:
        for f in sorted(glob.glob(os.path.join(base, pat), recursive=True)):
            if os.path.isfile(f) and "_smoke" not in os.path.basename(f):
                rel = os.path.relpath(f, base); out.append((f, f"{dest}/{rel}", os.path.getsize(f)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-repo", default="ZhangLab-DeepNeuroCogLab/SMoRe"); ap.add_argument("--data-repo", default="ZhangLab-DeepNeuroCogLab/SMoRe-data")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--private", action="store_true", help="create the repos private (make public later in the repo settings)")
    a = ap.parse_args()
    plan = [(a.model_repo, "model", collect(MODEL_FILES), MODEL_CARD), (a.data_repo, "dataset", collect(DATA_FILES), DATA_CARD)]
    for repo, kind, files, card in plan:
        print(f"== {repo} ({kind}): {len(files)} files, {sum(s for _, _, s in files) / 1e9:.2f} GB")
        for src, dst, size in files[:8]:
            print(f"   {src} -> {dst} ({size / 1e6:.1f} MB)")
        if len(files) > 8:
            print(f"   ... {len(files) - 8} more")
    if a.dry_run:
        return
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    for repo, kind, files, card in plan:
        create_repo(repo, repo_type=kind, exist_ok=True, private=a.private)
        existing = set(api.list_repo_files(repo, repo_type=kind))
        if "README.md" not in existing:
            api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type=kind)
        from huggingface_hub import CommitOperationAdd
        todo = [(src, dst) for src, dst, _ in files if dst not in existing]
        groups = {}
        for src, dst in todo:
            groups.setdefault(dst.rsplit("/", 1)[0], []).append((src, dst))
        print(f"   {len(todo)} files to upload in {len(groups)} commits ({len(files) - len(todo)} already present)")
        for folder, items in groups.items():
            for i in range(0, len(items), 500):                       # <= 500 files per commit
                ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for src, dst in items[i:i + 500]]
                api.create_commit(repo_id=repo, repo_type=kind, operations=ops, commit_message=f"add {folder} ({len(ops)} files)")
                print(f"   committed {folder}: {len(ops)} files")
    print("done")


if __name__ == "__main__":
    main()
