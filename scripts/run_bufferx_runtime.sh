#!/usr/bin/env bash
# Run BUFFER-X with its isolated operators and preserve container driver paths.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_venv="$task_root/outputs/envs/bufferx"
export CUDA_HOME="${BUFFERX_CUDA_HOME:-/tmp/worldcache_bufferx_cuda124}"
if [[ -x "$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc" ]]; then
  export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
else
  export CC="${CC:-$(command -v gcc)}"
  export CXX="${CXX:-$(command -v g++)}"
fi
export PATH="$task_venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="4"
task_packages="$("$task_venv/bin/python" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent.parent/"nvidia")')"
task_venv_packages="$("$task_venv/bin/python" -c 'import site; print(site.getsitepackages()[0]+"/nvidia")')"
task_includes="$(find "$task_packages" "$task_venv_packages" -mindepth 2 -maxdepth 2 -type d -name include 2>/dev/null | sort -u | paste -sd:)"
task_libraries="$(find "$task_packages" -mindepth 2 -maxdepth 2 -type d -name lib | sort | paste -sd:)"
export CPATH="$task_includes${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$task_libraries:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$task_venv/bin/python" "$@"
