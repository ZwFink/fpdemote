#!/bin/bash
# Builds tests/math_sanity.hip with Proteus and runs it in double, with
# demote-fp-math, and with both passes.
# Usage: run_math_sanity.sh <libDemoteFP.so> [build dir]
set -euo pipefail
PLUGIN=$(readlink -f "$1")
OUT=${2:-$(pwd)/math-sanity}
ROCM=${ROCM_PATH:-/opt/rocm-6.4.0}
PROTEUS=${PROTEUS_DIR:?set PROTEUS_DIR to the Proteus install}
ARCH=${OFFLOAD_ARCH:-gfx942}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT"
"$ROCM/llvm/bin/clang++" -O3 -std=c++17 -x hip --offload-arch="$ARCH" \
  -fplugin="$PROTEUS/lib64/libProteusPass.so" \
  -fpass-plugin="$PROTEUS/lib64/libProteusPass.so" \
  -Xclang -mllvm -Xclang -force-proteus-jit-annotate-all \
  -I"$PROTEUS/include" "$HERE/math_sanity.hip" -o "$OUT/math_sanity" \
  -L"$PROTEUS/lib64" -lproteus -Wl,-rpath,"$PROTEUS/lib64"

# The plugin only runs on Proteus's serial code generation path.
export PROTEUS_CODEGEN=serial PROTEUS_USE_STORED_CACHE=0
export PROTEUS_SPECIALIZE_ARGS=0 PROTEUS_SPECIALIZE_DIMS=0
export PROTEUS_SPECIALIZE_DIMS_RANGE=0

echo "== double (no plugin)"
"$OUT/math_sanity"
echo "== demote-fp-math"
FPDEMOTE_PLUGIN=$PLUGIN FPDEMOTE_PIPELINE=demote-fp-math "$OUT/math_sanity"
echo "== demote-fp,demote-fp-math"
FPDEMOTE_PLUGIN=$PLUGIN FPDEMOTE_PIPELINE=demote-fp,demote-fp-math \
  "$OUT/math_sanity"
