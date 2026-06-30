# Stage 3 — DN-Splatter environment (`.venv-splat`)

Hyper-realistic stage uses **DN-Splatter** (depth+normal-regularized 3DGS → mesh),
which needs **gsplat + nerfstudio**. gsplat's CUDA kernels will **not** build
against the main pipeline's torch (2.12.1+cu132) on Windows, so stage 3 lives in
a **separate, isolated venv** with torch built for **cu128** — which *matches the
installed nvcc 12.8 toolkit* and still supports the Blackwell sm_120 GPU. This
does **not** change the system CUDA; it's just a torch wheel that matches the
toolkit. The main pipeline venv (`.venv`, cu132) is untouched.

## Recipe (reproducible)

```bash
# 1. Isolated venv from Python 3.11.9 (same base as .venv)
"C:/Users/Tom-o/AppData/Local/Python/pythoncore-3.11-64/python.exe" -m venv .venv-splat
PY=".venv-splat/Scripts/python.exe"

# 2. torch cu128 (matches nvcc 12.8, supports sm_120 Blackwell)
$PY -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 3. DN-Splatter's tested combo (nerfstudio pins gsplat==1.0.0; that API is what
#    dn-splatter uses — `rasterize_gaussians`. Newer gsplat 1.x renamed it.)
$PY -m pip install "nerfstudio==1.1.3" "gsplat==1.0.0"

# 4. dn-splatter itself, WITHOUT deps (we keep the working torch/gsplat/nerfstudio)
git clone --depth 1 https://github.com/maturk/dn-splatter external/dn-splatter
$PY -m pip install "setuptools==69.5.1"
$PY -m pip install -e external/dn-splatter --no-deps

# 5. Transitive pure-python deps that --no-deps skipped (surfaced at import):
$PY -m pip install natsort pytorch-lightning geffnet timm omnidata-tools ninja
#   (PyMCubes / vdbfusion / rerun-sdk are only for ALTERNATE mesh paths; the
#    primary `gs-mesh o3dtsdf` uses open3d and does not need them.)
```

## Building the CUDA kernels (first run only)

gsplat JIT-compiles its kernels on first import and needs **MSVC `cl.exe`** on
PATH, which isn't there by default. Run any gsplat-touching command from a shell
that has sourced VS2022's `vcvars64.bat` once:

```
cmd /c "\"C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat\" && <python/ns-train command>"
```

After the first successful build the kernels are cached (under
`%LOCALAPPDATA%\...\torch_extensions\...\gsplat_cuda`) and subsequent runs don't
need `cl.exe`.

> Note: gsplat **1.5.x** additionally needs a one-line Windows patch (drop the
> GCC-only `-Wno-attributes` flag from `cuda/_backend.py`), but **1.0.0**
> (the version we use) already uses clean `-O3` cflags — no patch needed.

## Verified working
- torch 2.8.0+cu128, `cuda.is_available()`, GPU = RTX 5070, **sm_120**, fp16 OK
- gsplat 1.0.0 CUDA kernels build + load (`_C is not None`)
- nerfstudio 1.1.3 + dn-splatter import; `ns-train dn-splatter --help` OK

## Run the stage

```bash
# (main .venv) export our metric, posed, depth data -> nerfstudio dataset
python src/splat/export_nerfstudio.py
#   -> <session>/output/splat/nerfstudio/{images,depth,transforms.json,points3D.ply}

# (.venv-splat) train DN-Splatter, METRIC (orientation/center/scale OFF),
# normals derived from our depth, init from the LiDAR-anchored points3D.ply
ns-train dn-splatter \
  --pipeline.model.use-depth-loss True --pipeline.model.depth-lambda 0.2 \
  --pipeline.model.use-normal-loss True --pipeline.model.normal-supervision depth \
  --output-dir <session>/output/splat/runs \
  normal-nerfstudio --data <session>/output/splat/nerfstudio \
  --orientation-method none --center-method none --auto-scale-poses False

# (.venv-splat) export the mesh
gs-mesh o3dtsdf --load-config <run>/config.yml --output-dir <session>/output/splat/mesh
```
