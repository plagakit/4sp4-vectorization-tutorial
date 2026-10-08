#include "matrix_mult.h"

void matmult_base(const float* A, const float* B, float* C, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}

void matmult_ikj(const float* A, const float* B, float* C, int n)
{
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}

constexpr int s = 50; // should divide n, we assume it can

void matmult_tiled(const float* A, const float* B, float* C, int n)
{
    for (int oi = 0; oi < n; oi += s)
        for (int oj = 0; oj < n; oj += s)
            for (int ok = 0; ok < n; ok += s)
                for (int i = oi; i < oi+s; i++)
                    for (int j = oj; j < oj+s; j++)
                        for (int k = ok; k < ok+s; k++)
                            C[i*n + j] += A[i*n + k] * B[k*n + j];
}

void matmult_tiled_ikj(const float* A, const float* B, float* C, int n)
{
    for (int oi = 0; oi < n; oi += s)
        for (int ok = 0; ok < n; ok += s)
            for (int oj = 0; oj < n; oj += s)
                for (int i = oi; i < oi+s; i++)
                    for (int k = ok; k < ok+s; k++)
                        for (int j = oj; j < oj+s; j++)
                            C[i*n + j] += A[i*n + k] * B[k*n + j];
}