#!/usr/bin/env bash
# Build all V7 C++ targets.
#
# CUDA 13 + CMake 3.28 quirks on this box:
#   * nvcc isn't on PATH (it's at /usr/local/cuda/bin/nvcc)
#   * CUDA architecture auto-detection fails
# So we pass both explicitly. LIBTORCH and NVCC live in scripts/env_paths.cfg
# (with env_paths.cfg.local overrides) — fill them in once per server.
#
# Usage:
#   bash scripts/build.sh                     # configure (if needed) + build all
#   bash scripts/build.sh --target selfplay_main   # extra args go to cmake --build
#   LIBTORCH=/path/to/libtorch bash scripts/build.sh   # one-off override
#   CUDA_ARCH=75 bash scripts/build.sh
#   rm -rf cpp/build && bash scripts/build.sh # force fresh configure
#
# See SETUP.md for first-time setup.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
SRC_DIR="$ROOT/cpp"
BUILD_DIR="$SRC_DIR/build"

# Pull LIBTORCH / NVCC (and any .local overrides) from env_paths.cfg. Env vars
# set before invocation still win — env_paths.cfg uses ${VAR:-default} for each.
# Build is independent of experiment config (no CONFIG_DIR needed here).
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_paths.cfg"

if [[ ! -d "$LIBTORCH" ]]; then
    echo "[build.sh] ERROR: LIBTORCH=$LIBTORCH does not exist." >&2
    echo "[build.sh] Set it in scripts/env_paths.cfg.local, e.g.:" >&2
    echo "[build.sh]   LIBTORCH=\"\$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')\"" >&2
    exit 1
fi
if [[ ! -x "$NVCC" ]]; then
    echo "[build.sh] ERROR: NVCC=$NVCC not found or not executable." >&2
    echo "[build.sh] Install CUDA toolkit and set NVCC in scripts/env_paths.cfg.local." >&2
    exit 1
fi

# CUDA_ARCH: auto-detect from nvidia-smi if unset, so the same default works on
# every box (Ada → 89, Blackwell → 120, etc). Override via `CUDA_ARCH=NN bash
# scripts/build.sh` if you need a specific arch (e.g. cross-compiling).
if [[ -z "${CUDA_ARCH:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -1 | tr -d '.')
    fi
    CUDA_ARCH="${CUDA_ARCH:-89}"
fi

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "[build.sh] configure: BUILD_DIR=$BUILD_DIR LIBTORCH=$LIBTORCH NVCC=$NVCC CUDA_ARCH=$CUDA_ARCH"
    cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
        -DCMAKE_PREFIX_PATH="$LIBTORCH" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_COMPILER="$NVCC"
fi

echo "[build.sh] build"
cmake --build "$BUILD_DIR" -j "$@"
