#ifndef MATRIX_VEC_H
#define MATRIX_VEC_H

// Base unvectorized implementation
void matmul_base(const float* A, const float* x, float* y, int m, int n);

// Inner loop vectorization (SIMD on j loop)
void matmul_inner_vec(const float* A, const float* x, float* y, int m, int n);

// 2D vectorization (SIMD on both i and j)
void matmul_2d_vec(const float* A, const float* x, float* y, int m, int n);


void matmul_inner_unrolled(const float* A, const float* x, float* y, int m, int n);
void matmul_inner_alt(const float* A, const float* x, float* y, int m, int n);
void matmul_innervec_outerunrolled(const float* A, const float* x, float* y, int m, int n);
void matmul_vec_outer_vec(const float* A, const float* x, float* y, int m, int n);

#endif // MATRIX_VEC_H

