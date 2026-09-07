# smr_updates145 — renders verified (P49/P50 hit); gsplat did not build; crop the renders for framing

## 1. Scored
- **P49 hit:** 19,209 objects rendered (1 failure); checker excluded 262 (1.4 %: 260 EMPTY, 1 SATURATED, 1 FAILED)
  -> 18,947 usable training objects. **P50 hit:** GSO 1,033 objects, 0 excluded. 12 GB of renders.
- Object coverage is 11-13 % of the frame (median), not the 20-45 % predicted: the bounding-SPHERE normalisation is
  conservative (real objects fill a fraction of their sphere), so the object spans ~90 of 256 px and the backbone sees a
  small object in an upsampled 518^2 image. Fix without re-rendering: a centred 70 % crop resized back to 256^2
  (exactly a narrower camera: K scaled by 1/0.7, principal point unchanged, c2w unchanged). Done on disk because the
  backbones read image files. Coverage ~12 % -> ~24 %.

## 2. gsplat: `_C is None` -> the CUDA extension did not build/load
gsplat 1.5.3 JIT-compiles its kernels on first import and swallows the failure; the AttributeError appears at the
first rasterization call. Diagnose and fix with their prebuilt wheel for your torch/CUDA pair:
```bash
cd ~/smr && source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"; which nvcc; nvcc --version 2>/dev/null | tail -1
python - <<'EOF'
import torch, re
tv = re.match(r"(\d+)\.(\d+)", torch.__version__); cu = torch.version.cuda.replace(".", "")
print(f"pip uninstall -y gsplat && pip install gsplat --index-url https://docs.gsplat.studio/whl/pt{tv.group(1)}{tv.group(2)}cu{cu}")
