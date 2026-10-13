#!/usr/bin/env bash
# Build upstream CUDA operators in an isolated venv using the existing torch.
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_cuda_root="${1:-/tmp/worldcache_bufferx_cuda124}"
task_source_root="${2:-/tmp/worldcache_bufferx_build}"
if [[ $# -ge 3 ]]; then
  task_base_python="$3"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  task_base_python="$CONDA_PREFIX/bin/python"
else
  task_base_python="$(command -v python3 || command -v python)"
fi
task_venv="$task_root/outputs/envs/bufferx"
task_log_root="$task_root/outputs/reports/20261008_bufferx_preflight"
mkdir -p "$task_log_root"

if [[ ! -x "$task_venv/bin/python" ]]; then
  "$task_base_python" -m venv --system-site-packages "$task_venv"
fi
export CUDA_HOME="$task_cuda_root"
export CC="$task_cuda_root/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$task_cuda_root/bin/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="4"
export PATH="$task_venv/bin:$PATH"

# PyTorch's pip CUDA packages already provide the remaining development
# headers/libraries; avoid downloading another full CUDA library suite.
task_cuda_includes="$("$task_venv/bin/python" -c 'from pathlib import Path; import torch; p=Path(torch.__file__).resolve().parent.parent/"nvidia"; print(":".join(str(d) for d in sorted(p.glob("*/include"))))')"
export CPATH="$task_cuda_includes${CPATH:+:$CPATH}"
task_cuda_libraries="$("$task_venv/bin/python" -c 'from pathlib import Path; import torch; p=Path(torch.__file__).resolve().parent.parent/"nvidia"; print(":".join(str(d) for d in sorted(p.glob("*/lib"))))')"
export LD_LIBRARY_PATH="$task_cuda_libraries:$task_cuda_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$task_venv/bin/python" - "$task_cuda_root" <<'PY'
from pathlib import Path
import sys
import torch
cuda = Path(sys.argv[1])
packages = Path(torch.__file__).resolve().parent.parent / "nvidia"
for library, package in (("cublas", "cublas"), ("cusolver", "cusolver")):
    candidates = sorted((packages / package / "lib").glob(f"lib{library}.so.*"))
    if not candidates:
        raise FileNotFoundError(f"PyTorch CUDA package library missing: {library}")
    target = cuda / "lib" / f"lib{library}.so"
    if not target.exists():
        target.symlink_to(candidates[0])
PY

"$task_venv/bin/python" -m pip install --quiet --no-cache-dir easydict kornia nibabel ninja
"$task_venv/bin/python" -m pip install --no-deps --no-build-isolation --no-cache-dir \
  "$task_source_root/Pointnet2_PyTorch/pointnet2_ops_lib" > "$task_log_root/pointnet_build.log" 2>&1
"$task_venv/bin/python" -m pip install --no-deps --no-build-isolation --no-cache-dir \
  "$task_source_root/knn_cuda" > "$task_log_root/knn_install.log" 2>&1
"$task_venv/bin/python" -m pip install --no-deps --no-build-isolation --no-cache-dir \
  "$task_source_root/torch-batch-svd" > "$task_log_root/svd_build.log" 2>&1

# Importing knn_cuda also compiles its upstream JIT operator.
"$task_venv/bin/python" - <<'PY' > "$task_log_root/operator_imports.json"
import json
import pointnet2_ops.pointnet2_utils
from knn_cuda import KNN
from torch_batch_svd import svd
import torch
print(json.dumps({"status": "pass", "torch": torch.__version__, "cuda": torch.version.cuda,
                  "operators": ["pointnet2_ops", "knn_cuda", "torch_batch_svd"]}, indent=2))
PY
