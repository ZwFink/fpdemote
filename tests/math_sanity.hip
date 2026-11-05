// Sanity check for demote-fp-math: evaluates each math function in a JIT
// compiled kernel and reports its maximum error against the host's double
// result. Demoted functions should show float-level error (about 1e-7) and
// functions the pass keeps in double should show double-level error.
//
// The plugin is registered from FPDEMOTE_PLUGIN and FPDEMOTE_PIPELINE; with
// neither set the kernel runs in double.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <hip/hip_runtime.h>
#include <proteus/Init.h>

#define CHECK(X)                                                               \
  do {                                                                         \
    if ((X) != hipSuccess) {                                                   \
      std::fprintf(stderr, "%s failed\n", #X);                                 \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

enum Fn {
  Sin, Cos, Log, Log2, Log10, Exp, Exp2, Exp10, Pow, Sqrt, Fabs, Floor, Ceil, Trunc,
  Rint, Round, Copysign, Fma, Fmin, Fmax, Ldexp, Tan, Atan2, NumFns
};

const char *Names[NumFns] = {
    "sin",   "cos",  "log",   "log2", "log10", "exp",      "exp2", "exp10", "pow",
    "sqrt",  "fabs", "floor", "ceil", "trunc", "rint",     "round", "copysign",
    "fma",   "fmin", "fmax",  "ldexp", "tan",  "atan2"};

// Functions the pass is expected to leave in double.
bool isKeptInDouble(int F) { return F == Pow || F == Tan || F == Atan2; }

__host__ __device__ double evaluate(int F, double X, double Y) {
  switch (F) {
  case Sin: return sin(X);
  case Cos: return cos(X);
  case Log: return log(X);
  case Log2: return log2(X);
  case Log10: return log10(X);
  case Exp: return exp(Y);
  case Exp2: return exp2(Y);
  case Exp10: return exp10(Y);
  case Pow: return pow(X, Y);
  case Sqrt: return sqrt(X);
  case Fabs: return fabs(-X);
  case Floor: return floor(X);
  case Ceil: return ceil(X);
  case Trunc: return trunc(-X);
  case Rint: return rint(X);
  case Round: return round(X);
  case Copysign: return copysign(X, Y);
  case Fma: return fma(X, Y, X);
  case Fmin: return fmin(X, Y);
  case Fmax: return fmax(X, Y);
  case Ldexp: return ldexp(X, 5);
  case Tan: return tan(X / 8.0);
  case Atan2: return atan2(Y, X);
  }
  return 0.0;
}

__global__ void mathKernel(const double *X, const double *Y, double *Out,
                           int N) {
  int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  for (int F = 0; F < NumFns; ++F)
    Out[F * N + I] = evaluate(F, X[I], Y[I]);
}

int main() {
  const char *Plugin = std::getenv("FPDEMOTE_PLUGIN");
  const char *Pipeline = std::getenv("FPDEMOTE_PIPELINE");
  if (Plugin && Pipeline)
    proteus::registerJITPassPlugin(Plugin, Pipeline,
                                   proteus::JITPassPluginPosition::Prepend);

  // X stays in (0.1, 2*pi) and away from integers so rounding functions do
  // not flip when X is rounded to float; Y covers [-3, 3].
  const int N = 4096;
  std::vector<double> X(N), Y(N), Out(NumFns * N);
  for (int I = 0; I < N; ++I) {
    X[I] = 0.1 + std::floor(6.0 * I / N) + 0.25 + 0.5 * (I % 97) / 97.0;
    Y[I] = -3.0 + 6.0 * ((I * 37) % N) / N + 1e-3;
  }

  double *DX, *DY, *DOut;
  CHECK(hipMalloc(&DX, N * sizeof(double)));
  CHECK(hipMalloc(&DY, N * sizeof(double)));
  CHECK(hipMalloc(&DOut, NumFns * N * sizeof(double)));
  CHECK(hipMemcpy(DX, X.data(), N * sizeof(double), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(DY, Y.data(), N * sizeof(double), hipMemcpyHostToDevice));
  mathKernel<<<(N + 255) / 256, 256>>>(DX, DY, DOut, N);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(Out.data(), DOut, NumFns * N * sizeof(double),
                  hipMemcpyDeviceToHost));

  std::printf("%-9s %12s  %s\n", "function", "max error", "precision");
  for (int F = 0; F < NumFns; ++F) {
    double MaxErr = 0.0;
    for (int I = 0; I < N; ++I) {
      double Ref = evaluate(F, X[I], Y[I]);
      double Err = std::fabs(Out[F * N + I] - Ref) /
                   std::fmax(std::fabs(Ref), 1.0);
      MaxErr = std::fmax(MaxErr, Err);
    }
    const char *Class = MaxErr < 1e-12  ? "double"
                        : MaxErr < 1e-5 ? "float"
                                        : "WRONG";
    std::printf("%-9s %12.3e  %s%s\n", Names[F], MaxErr, Class,
                isKeptInDouble(F) ? "  (call kept in double by demote-fp-math)"
                                  : "");
  }
  CHECK(hipFree(DX));
  CHECK(hipFree(DY));
  CHECK(hipFree(DOut));
  return 0;
}
