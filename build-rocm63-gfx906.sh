#!/bin/bash
# Build script for llama.cpp with ROCm 6.3 + gfx906 support
# This script fixes the compilation issues with ROCm 6.3's clang 18

set -e

BUILD_DIR="${BUILD_DIR:-/tmp/llama-build}"
SRC_DIR="${SRC_DIR:-/tmp/llama-src}"

# ROCm paths
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-6.3.0}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export CMAKE_PREFIX_PATH="$ROCM_PATH/lib/cmake"

# Fix for ROCm 6.3 + gfx906: clang 18 needs C++ include paths
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"

# Compiler settings
export HIPCXX="$ROCM_PATH/lib/llvm/bin/clang++"

echo "=== Building llama.cpp for ROCm 6.3 + gfx906 ==="
echo "ROCm Path: $ROCM_PATH"
echo "Build Dir: $BUILD_DIR"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with HIP support
cmake "$SRC_DIR" -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-fPIC -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11" \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH/lib/cmake"

# Build with parallel jobs
cmake --build build --parallel $(nproc)

echo "=== Build complete ==="
echo "Binary location: $BUILD_DIR/build/bin/llama-cli"
