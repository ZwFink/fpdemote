// Sanity check for the atomics part of demote-fp. Every check has an answer that does not
// depend on the order in which threads update the accumulator:
//
//   count     N threads each add 1.0 to 0. The sum is N and the returned old
//             values are exactly 0 .. N-1, so no update is lost and every
//             thread gets the value its update replaced.
//   saturate  N threads each add 1.0 to 2^24. A double accumulator reaches
//             2^24 + N; a float one stays at 2^24, because 2^24 + 1 rounds
//             back to 2^24.
//   subtract  N threads each subtract 1.0 from 2^25 with fsub. A double
//             accumulator reaches 2^25 - N; in float 2^25 - 1 rounds back to
//             2^25.
//
// N is small because every thread contends for one address, and the cmpxchg
// loop's retries grow with the square of the number of contending threads.
//   max       An fmax atomic, which the pass keeps in double.
//
// The plugin is registered from FPDEMOTE_PLUGIN and FPDEMOTE_PIPELINE. The
// expected results are the float ones when the pipeline is demote-fp without
// no-atomics and the double ones otherwise. Exits non-zero on any mismatch.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__global__ void countKernel(double *Sum, double *Old, int N) {
  int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    Old[I] = atomicAdd(Sum, 1.0);
}

__global__ void saturateKernel(double *Sum, int N) {
  int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    atomicAdd(Sum, 1.0);
}

__global__ void subtractKernel(double *Sum, int N) {
  int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    __hip_atomic_fetch_sub(Sum, 1.0, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
}

__global__ void maxKernel(double *Max, int N) {
  int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    __hip_atomic_fetch_max(Max, static_cast<double>(I), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
}

int Failures = 0;

void expect(const char *Name, double Got, double Want) {
  bool Ok = Got == Want;
  std::printf("%-9s got %.17g, want %.17g  %s\n", Name, Got, Want,
              Ok ? "ok" : "FAIL");
  Failures += !Ok;
}

double runOnScalar(double Init, void (*Launch)(double *)) {
  double *D;
  CHECK(hipMalloc(&D, sizeof(double)));
  CHECK(hipMemcpy(D, &Init, sizeof(double), hipMemcpyHostToDevice));
  Launch(D);
  CHECK(hipGetLastError());
  double Out;
  CHECK(hipMemcpy(&Out, D, sizeof(double), hipMemcpyDeviceToHost));
  CHECK(hipFree(D));
  return Out;
}

int blocks(int N) { return (N + 255) / 256; }

int main() {
  const char *Plugin = std::getenv("FPDEMOTE_PLUGIN");
  const char *Pipeline = std::getenv("FPDEMOTE_PIPELINE");
  if (Plugin && Pipeline)
    proteus::registerJITPassPlugin(Plugin, Pipeline,
                                   proteus::JITPassPluginPosition::Prepend);
  bool FloatAtomics = Plugin && Pipeline &&
                      std::strstr(Pipeline, "demote-fp") &&
                      !std::strstr(Pipeline, "no-atomics");
  std::printf("expecting %s accumulation\n", FloatAtomics ? "float" : "double");

  const int Count = 1 << 12;
  double *DSum, *DOld;
  CHECK(hipMalloc(&DSum, sizeof(double)));
  CHECK(hipMalloc(&DOld, Count * sizeof(double)));
  CHECK(hipMemset(DSum, 0, sizeof(double)));
  countKernel<<<blocks(Count), 256>>>(DSum, DOld, Count);
  CHECK(hipGetLastError());
  double Sum;
  std::vector<double> Old(Count);
  CHECK(hipMemcpy(&Sum, DSum, sizeof(double), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(Old.data(), DOld, Count * sizeof(double),
                  hipMemcpyDeviceToHost));
  CHECK(hipFree(DSum));
  CHECK(hipFree(DOld));
  expect("count", Sum, Count);
  std::sort(Old.begin(), Old.end());
  int Misplaced = 0;
  for (int I = 0; I < Count; ++I)
    Misplaced += Old[I] != I;
  expect("returned", Misplaced, 0);

  const double P24 = 1 << 24, P25 = 1 << 25;
  expect("saturate",
         runOnScalar(P24, [](double *D) {
           saturateKernel<<<blocks(1 << 12), 256>>>(D, 1 << 12);
         }),
         FloatAtomics ? P24 : P24 + Count);
  expect("subtract",
         runOnScalar(P25, [](double *D) {
           subtractKernel<<<blocks(1 << 12), 256>>>(D, 1 << 12);
         }),
         FloatAtomics ? P25 : P25 - Count);
  expect("max",
         runOnScalar(0.0, [](double *D) {
           maxKernel<<<blocks(1 << 12), 256>>>(D, 1 << 12);
         }),
         Count - 1);

  std::printf("%s\n", Failures ? "FAILED" : "PASSED");
  return Failures ? 1 : 0;
}
