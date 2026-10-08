#include <benchmark/benchmark.h>
#include <vector>
#include <iostream>
#include "matrix_mult.h"

#define N 1000
#define RUNS 20

typedef void (*MatmulFunc)(const float*, const float*, float*, int);

void test_matmult(
    std::string name, MatmulFunc f, 
    const std::vector<float>& A, const std::vector<float>& B, int n, 
    const std::vector<float>& expected
)  {
    std::cout << "Testing \"" << name << "\"...\n";
    std::vector<float> result;
    result.resize(expected.size());
    for (int i = 0; i < n*n; i++) { result[i] = 0.0f; }

    f(A.data(), B.data(), result.data(), n);

    for (int i = 0; i < expected.size(); i++) {
        auto diff = std::abs(result[i] - expected[i]);
        if (diff > 1.0) {
            std::cerr << name << " failed! " << result[i] << " != " << expected[i] << " @ " << i << "\n";
            return;
        }
    }
}

#define DEFINE_BENCHMARK(func) \
    static void BM_##func##_(benchmark::State& state) { \
        int n = state.range(0); \
        std::vector<float> A(n * n), B(n * n), C(n * n); \
        for (int i = 0; i < n * n; i++) { \
            A[i] = 1.5f; B[i] = 2.0f; C[i] = 0.0f; \
        } \
        for (auto _ : state) { \
            func(A.data(), B.data(), C.data(), n); \
        } \
        state.counters["FLOPs"] = benchmark::Counter(2.0 * n * n, benchmark::Counter::kIsRate); \
    } \
    BENCHMARK(BM_##func##_)->Args({N})->Iterations(1)->Repetitions(RUNS);

DEFINE_BENCHMARK(matmult_base)
DEFINE_BENCHMARK(matmult_ikj)
DEFINE_BENCHMARK(matmult_tiled)
DEFINE_BENCHMARK(matmult_tiled_ikj)

int main(int argc, char** argv) { 
    char arg0_default[] = "benchmark"; 
    char* args_default = arg0_default; 
    if (!argv) { 
        argc = 1; 
        argv = &args_default; 
    } 

    std::cout << "Testing matrix multiplication functions..." << std::endl;
    std::vector<float> A(N*N), B(N*N), C(N*N);
    for (int i = 0; i < N * N; i++) {
        float r = 1.2f;//static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        A[i] = r;
        B[i] = r;
        C[i] = 0.0f;
    }
    matmult_base(A.data(), B.data(), C.data(), N);

    test_matmult("base", matmult_base, A, B, N, C);
    test_matmult("ikj", matmult_ikj, A, B, N, C);
    test_matmult("tiled", matmult_tiled, A, B, N, C);
    test_matmult("tiled ikj", matmult_tiled_ikj, A, B, N, C);

    ::benchmark::Initialize(&argc, argv); 
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) 
        return 1; 
    ::benchmark::RunSpecifiedBenchmarks(); 
    ::benchmark::Shutdown(); 
    return 0; 
} int main(int, char**);
