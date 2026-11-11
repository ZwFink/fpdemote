# Shared setup for the sanity tests: sourced with the plugin path and build
# directory as $1 and $2.
set -euo pipefail
PLUGIN=$(readlink -f "$1")
OUT=${2:-$(pwd)/sanity}
ROCM=${ROCM_PATH:-/opt/rocm-6.4.0}
PROTEUS=${PROTEUS_DIR:?set PROTEUS_DIR to the Proteus install}
ARCH=${OFFLOAD_ARCH:-gfx942}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT"

# build_sanity <name>: compiles tests/<name>.hip into $OUT/<name> with Proteus.
build_sanity() {
  "$ROCM/llvm/bin/clang++" -O3 -std=c++17 -x hip --offload-arch="$ARCH" \
    -fplugin="$PROTEUS/lib64/libProteusPass.so" \
    -fpass-plugin="$PROTEUS/lib64/libProteusPass.so" \
    -Xclang -mllvm -Xclang -force-proteus-jit-annotate-all \
    -I"$PROTEUS/include" "$HERE/$1.hip" -o "$OUT/$1" \
    -L"$PROTEUS/lib64" -lproteus -Wl,-rpath,"$PROTEUS/lib64"
}

# The plugin only runs on Proteus's serial code generation path.
export PROTEUS_CODEGEN=serial PROTEUS_USE_STORED_CACHE=0
export PROTEUS_SPECIALIZE_ARGS=0 PROTEUS_SPECIALIZE_DIMS=0
export PROTEUS_SPECIALIZE_DIMS_RANGE=0
