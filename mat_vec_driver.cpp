#include <benchmark/benchmark.h>
#include <vector>
#include "matrix_vec.h"

#define M 10000
#define N 10000
#define RUNS 100

#define DEFINE_BENCHMARK(func) \
    static void BM_##func##_(benchmark::State& state) { \
        int m = state.range(0); \
        int n = state.range(1); \
        std::vector<float> A(m * n), x(n), y(m); \
        for (int i = 0; i < m * n; i++) A[i] = 1.5f; \
        for (int i = 0; i < n; i++) x[i] = 2.0f; \
        for (auto _ : state) { \
            func(A.data(), x.data(), y.data(), m, n); \
        } \
        state.counters["FLOPs"] = benchmark::Counter(2.0 * m * n, benchmark::Counter::kIsRate); \
    } \
    BENCHMARK(BM_##func##_)->Args({M, N})->Iterations(1)->Repetitions(RUNS);

DEFINE_BENCHMARK(matmul_base)
DEFINE_BENCHMARK(matmul_inner_unrolled)
DEFINE_BENCHMARK(matmul_inner_vec)
DEFINE_BENCHMARK(matmul_inner_alt)
DEFINE_BENCHMARK(matmul_innervec_outerunrolled)

BENCHMARK_MAIN();
