#include <benchmark/benchmark.h>
#include <vector>
#include <iostream>
#include "matrix_vec.h"

#define M 10000
#define N 10000
#define RUNS 100

typedef void (*MatmulFunc)(const float*, const float*, float*, int, int);

void test_matmul(std::string name, MatmulFunc f, const std::vector<float>& A, const std::vector<float>& x, int m, int n, const std::vector<float>& expected)  {
    std::vector<float> result;
    result.resize(expected.size());
    f(A.data(), x.data(), result.data(), m, n);
    for (int i = 0; i < expected.size(); i++) {
        auto diff = std::abs(result[i] - expected[i]);
        if (diff > 10.0) {
            std::cerr << name << " failed! " << result[i] << " != " << expected[i] << " @ " << i << "\n";
            return;
        }
    }
}

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
// DEFINE_BENCHMARK(matmul_inner_unrolled)
DEFINE_BENCHMARK(matmul_inner_vec)
// DEFINE_BENCHMARK(matmul_inner_alt)
DEFINE_BENCHMARK(matmul_innervec_outerunrolled)
DEFINE_BENCHMARK(matmul_vec_outer_vec)
DEFINE_BENCHMARK(matmul_2d_vec)
DEFINE_BENCHMARK(matmul_2d_vec_4rows)

int main(int argc, char** argv) { 
    char arg0_default[] = "benchmark"; 
    char* args_default = arg0_default; 
    if (!argv) { 
        argc = 1; 
        argv = &args_default; 
    } 

    // test funcs, not using gtest but its ok
    std::vector<float> A(M * N), x(N), y(M); \
    for (int i = 0; i < M * N; i++) {
        float r = 1.2f;//static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        A[i] = r;
    }
    for (int i = 0; i < N; i++) {
        float r = 2.0f;//static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        x[i] = r;
        y[i] = 0.0f;
    }
    matmul_base(A.data(), x.data(), y.data(), M, N);

    test_matmul("base", matmul_base, A, x, M, N, y);
    test_matmul("inner unrolled", matmul_inner_unrolled, A, x, M, N, y);
    test_matmul("inner vec", matmul_inner_vec, A, x, M, N, y);
    test_matmul("inner alt", matmul_inner_alt, A, x, M, N, y);
    test_matmul("inner vec + outer unrolled", matmul_innervec_outerunrolled, A, x, M, N, y);
    test_matmul("outer vec", matmul_vec_outer_vec, A, x, M, N, y);
    test_matmul("2d", matmul_2d_vec, A, x, M, N, y);
    test_matmul("2d 4rows", matmul_2d_vec_4rows, A, x, M, N, y);

    ::benchmark::Initialize(&argc, argv); 
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) 
        return 1; 
    ::benchmark::RunSpecifiedBenchmarks(); 
    ::benchmark::Shutdown(); 
    return 0; 
} int main(int, char**);
