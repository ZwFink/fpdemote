# fpdemote

A [Proteus](https://github.com/Olympus-HPC/proteus) JIT pass plugin that keeps
`double` storage but performs `double` computation in `float`.

Every rewrite has the same shape:

```
fptrunc operands to float -> float operation -> fpext result to double
```

Loads, stores and function signatures stay in `double`. Proteus's O3 pipeline
then removes the back-to-back `fpext`/`fptrunc` pairs between chained
operations. Atomic accumulation is the exception to this shape; see
[Atomics](#atomics).

The plugin provides three module passes, which can be used alone or together:

| Pass | Demotes |
|---|---|
| `demote-fp` | `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg`, `fcmp`, and `llvm.{sqrt,fabs,fma,fmuladd}.f64` |
| `demote-fp-math` | calls to the AMD math library (`__ocml_<fn>_f64`) and the matching `llvm.<fn>.f64` intrinsics, replaced by `llvm.<fn>.f32` intrinsics |
| `demote-fp-atomics` | `atomicrmw fadd` and `fsub` on `double`, replaced by a `cmpxchg` loop that adds in `float` |

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

proteus::registerJITPassPlugin("/path/to/libDemoteFP.so",
                               "demote-fp,demote-fp-math,demote-fp-atomics",
                               proteus::JITPassPluginPosition::Prepend);
```

The pipeline string selects the passes, in any combination: `demote-fp` for
arithmetic, `demote-fp-math` for math functions and `demote-fp-atomics` for
atomic accumulation.

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
| `QS_DEMOTE_MATH_SKIP` | Comma-separated math function names, such as `sin,log`, that `demote-fp-math` leaves in `double`. |
| `QS_DEMOTE_VERBOSE=1` | Print each demoted function or call to stderr. |

Functions whose names start with `__ocml` or `__ockl` (the AMD device
libraries) and functions containing `rngSample` are never rewritten, so the
math library's own `double` algorithms are not corrupted and the random-number
stream is unchanged.

`QS_DEMOTE_ONLY` and `QS_DEMOTE_SKIP` match the function that contains the
instruction or call. For math calls this is usually HIP's header wrapper, such
as `_ZL3logd` for `log(double)`, not the application function that calls
`log`, because nothing is inlined at `Prepend`. To choose which math functions
are demoted, use `QS_DEMOTE_MATH_SKIP`. Likewise, HIP's `atomicAdd(double *,
double)` is the wrapper `_Z9atomicAddPdd`, so a `double` `atomicAdd` anywhere in
the module is selected or skipped by that name.

Each pass prints a summary line per module:

```
[DemoteFP] module : 25 functions, arith=220 cmp=87 intr=2
[DemoteFPMath] module : demoted cos=1 log=2 sin=1; kept in double atan2=1
```

## Math functions

`demote-fp-math` only uses `f32` intrinsics that the AMDGPU backend (LLVM 19,
gfx942) lowers to plain instructions, without a math library:

| Demoted to an `f32` intrinsic | Lowering |
|---|---|
| `sin`, `cos` | hardware `v_sin_f32`/`v_cos_f32` of `x / 2π` |
| `log`, `log2`, `log10`, `exp`, `exp2`, `exp10` | hardware `v_log_f32`/`v_exp_f32` with range and accuracy fix-ups |
| `sqrt` | `v_sqrt_f32` with a correction step |
| `fabs`, `copysign`, `fmin`, `fmax`, `fma`, `ldexp` | single instructions |
| `floor`, `ceil`, `trunc`, `rint`, `nearbyint`, `round`, `roundeven` | single instructions |

These stay in `double`, and the summary line lists them under "kept in
double":

- `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh` and `atan2`: the
  backend has no lowering for the `f32` intrinsic and would need a library
  call.
- `pow`, `pown` and `powi`: the backend lowers the `f32` intrinsic to
  `exp2(y * log2(x))`, which returns NaN for a negative base and for
  `pow(0, 0)`.
- Any other math function not in the table above.

`v_sin_f32` and `v_cos_f32` do no full range reduction, so their error grows
with `|x|`. The sanity test below only covers `x` in `(0.1, 2π)`.

## Atomics

An atomic add happens in memory, so the operand-truncation shape above cannot
make the accumulator itself `float`: it would only round the added value. Instead
`demote-fp-atomics` replaces each `atomicrmw fadd` or `fsub` on `double` with a
compare-and-swap loop:

```
loop: old = load double
      new = fpext(fptrunc(old) + fptrunc(val))
      cmpxchg ptr, old, new   ; on the i64 bits, retried until it succeeds
```

The value in memory stays `double` but is always the `float`-rounded sum, so
the accumulator behaves exactly like a `float` accumulator that started from
the same value. The atomic's ordering, sync scope, alignment and volatility
carry over to the `cmpxchg`, and it still returns the value it replaced.
Other floating-point atomics (`fmax`, `fmin`) are left in `double` and listed
under "kept in double".

The loop is much slower than a hardware atomic when many threads update one
address, since every thread that loses the race retries. Updates spread over
many addresses, such as Quicksilver's tallies, are unaffected in practice.

## Testing

`tests/run_math_sanity.sh` builds a small Proteus HIP program that evaluates
every function in the table, plus `pow`, `tan` and `atan2`, and reports each
one's maximum error against the host's `double` result. It runs the kernel
without the plugin, with `demote-fp-math`, and with both passes:

```bash
PROTEUS_DIR=/path/to/proteus/install tests/run_math_sanity.sh build/libDemoteFP.so
```

Demoted functions should report `float` (errors of about 1e-7), functions the
pass keeps in `double` should report `double`, and nothing should report
`WRONG`. Rounding functions report `double` because their inputs are chosen
away from integers, so rounding the input to `float` does not change the
result.

`tests/run_atomic_sanity.sh` takes the same arguments and checks
`demote-fp-atomics` with results that do not depend on thread order: thousands
of threads add 1.0 to one value, which must lose no update and return each old
value exactly once; adding 1.0 to 2^24 and subtracting 1.0 from 2^25 must
leave the value unchanged in `float` and move it in `double`; and an `fmax`
atomic must stay in `double`. Each run exits non-zero on a mismatch.

## Why the plugin links its own LLVM

A shared Proteus built against a static LLVM (every ROCm 5.x–7.2 LLVM is
static) hides its LLVM symbols, so a plugin cannot resolve LLVM from the host
as an `opt` plugin would. The plugin therefore links its own copy and exports
only `llvmGetPassPluginInfo`, the same arrangement as Proteus's own
`JITTestPass`.

Two copies of LLVM in one process means static objects are not shared. In
particular, `APFloat` semantics are compared by address, so constant folding
inside the plugin misreads Proteus's floating-point constants (this hung the
JIT in testing). The passes therefore build all casts with
`IRBuilder<NoFolder>` and never create or inspect floating-point constants,
leaving folding to Proteus's O3. `demote-fp-math` only creates calls to
intrinsic declarations; any constants the lowering needs are created later by
Proteus's own LLVM. Plugin code that queries LLVM analyses through the analysis
manager is likely affected in the same way.

With a Proteus built against a shared `libLLVM.so`, the plugin links that
library instead and these restrictions do not apply.

## Status

Tested with Quicksilver (HIP, MI300A, ROCm 6.4.0, LLVM 19) on CORAL2 Problem 1
with 10,000 particles:

| Pipeline | Rewritten | Largest change in a summed tally | Per-cycle flux RMS change |
|---|---|---|---|
| `demote-fp` | 25 functions: 220 arithmetic, 87 compares, 2 intrinsics | 0.023% | 0.19% |
| `demote-fp-math` | `log` ×2, `sin`, `cos`, `sqrt` ×2 | 0.026% | 0.15% |
| `demote-fp,demote-fp-math` | both | 0.078% | 0.21% |
| different seeds, `double` | — | 0.25–0.64% | 2.7–3.7% |

All Quicksilver correctness checks pass in every configuration, and no
`__ocml_*_f64` code remains in the optimized kernel when `demote-fp-math` runs.
These are sanity checks of the implementation, not a fidelity study. At 1M
particles `demote-fp` alone changes tallies by 10–20× less than seed-to-seed
noise. There is no speedup from `demote-fp` on MI300A, where unpacked FP64 and
FP32 VALU throughput are equal; `demote-fp-math` has not been timed.

`demote-fp-atomics` rewrites Quicksilver's one `double` atomic, the scalar
flux tally in `MC_Segment_Outcome`, and does not change the particle counts.
With 200,000 particles, adding it to `demote-fp,demote-fp-math` changes
individual flux bins by about 5e-8 (median) with the default 230 energy groups
and about 1e-6 with 4 groups, and the summed flux by 3e-9 and 2e-7. The rest of
the demotion changes the summed flux by 1e-6 to 2e-5. Bins are summed in
`double` on the host, so per-bin rounding errors mostly cancel. Run time did
not change measurably.
