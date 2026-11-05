# fpdemote

A [Proteus](https://github.com/Olympus-HPC/proteus) JIT pass plugin that keeps
`double` storage but performs `double` arithmetic in `float`.

Each `double` `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg`, `fcmp` and each
call to `llvm.{sqrt,fabs,fma,fmuladd}.f64` is rewritten as

```
fptrunc operands to float -> float operation -> fpext result to double
```

Loads, stores, function signatures and the math library stay in `double`.
Proteus's O3 pipeline then removes the back-to-back `fpext`/`fptrunc` pairs
between chained operations.

## Building

Build against the same LLVM that Proteus was built with:

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.0/llvm/bin/clang++ \
  -DLLVM_DIR=/opt/rocm-6.4.0/llvm/lib/cmake/llvm
cmake --build build
```

This produces `build/libDemoteFP.so`.

## Using it with Proteus

Register the plugin before the first JIT compilation. It is meant to run at
`Prepend`, before Proteus's optimization pipeline, when no inlining has
happened yet and per-function selection by name still works:

```cpp
#include <proteus/Init.h>

proteus::registerJITPassPlugin("/path/to/libDemoteFP.so", "demote-fp",
                               proteus::JITPassPluginPosition::Prepend);
```

Proteus must run its plugin-capable code generation path:

```bash
export PROTEUS_CODEGEN=serial        # the default rtc path skips plugins
export PROTEUS_USE_STORED_CACHE=0    # do not reuse kernels compiled without the plugin
```

## Controls

| Variable | Effect |
|---|---|
| `QS_DEMOTE_ONLY` | Comma-separated substrings of mangled function names; only matching functions are demoted. |
| `QS_DEMOTE_SKIP` | Comma-separated substrings of function names to leave in `double`. |
| `QS_DEMOTE_VERBOSE=1` | Print per-function counts to stderr. |

Functions whose names start with `__ocml` or `__ockl` (the AMD device
libraries) and functions containing `rngSample` are never demoted, so the math
library keeps its own `double` algorithms and the random-number stream is
unchanged.

The pass always prints a summary line per module:

```
[DemoteFP] module : 25 functions, arith=220 cmp=87 intr=2
```

## Why the plugin links its own LLVM

A shared Proteus built against a static LLVM (every ROCm 5.x–7.2 LLVM is
static) hides its LLVM symbols, so a plugin cannot resolve LLVM from the host
as an `opt` plugin would. The plugin therefore links its own copy and exports
only `llvmGetPassPluginInfo`, the same arrangement as Proteus's own
`JITTestPass`.

Two copies of LLVM in one process means static objects are not shared. In
particular, `APFloat` semantics are compared by address, so constant folding
inside the plugin misreads Proteus's floating-point constants (this hung the
JIT in testing). The pass therefore builds all casts with `IRBuilder<NoFolder>`
and never creates or inspects floating-point constants, leaving folding to
Proteus's O3. Plugin code that queries LLVM analyses through the analysis
manager is likely affected in the same way.

With a Proteus built against a shared `libLLVM.so`, the plugin links that
library instead and these restrictions do not apply.

## Status

Tested with Quicksilver (HIP, MI300A, ROCm 6.4.0, LLVM 19): 25 device
functions demoted, all Quicksilver correctness checks pass, and the change in
tallies is 10–20× below seed-to-seed Monte Carlo noise. No speedup on MI300A,
where unpacked FP64 and FP32 VALU throughput are equal.
