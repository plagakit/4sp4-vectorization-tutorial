#include <immintrin.h>
#include "matrix_vec.h"

// Base unvectorized implementation
void matmul_base(const float* A, const float* x, float* y, int m, int n) {
    for (int i = 0; i < m; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

void matmul_inner_unrolled(const float* A, const float* x, float* y, int m, int n) {
    for (int row = 0; row < m; row++) {
        const float* rowaddr = &A[row*n];
        y[row] = 0.0f;

        int rem = n - n % 8;
        for (int col = 0; col < rem; col += 8) {
            y[row] += rowaddr[col] * x[col];
            y[row] += rowaddr[col+1] * x[col+1];
            y[row] += rowaddr[col+2] * x[col+2];
            y[row] += rowaddr[col+3] * x[col+3];
            y[row] += rowaddr[col+4] * x[col+4];
            y[row] += rowaddr[col+5] * x[col+5];
            y[row] += rowaddr[col+6] * x[col+6];
            y[row] += rowaddr[col+7] * x[col+7];
        }
        for (int col = rem; col < n; col++) {
            y[row] += rowaddr[col] * x[col];
        }
    }
}



// Inner loop vectorization (SIMD on j loop)
void matmul_inner_vec(const float* A, const float* x, float* y, int m, int n) {
    for (int i = 0; i < m; i++) {
        __m256 sum = _mm256_setzero_ps();
        int j = 0;

        // Process 8 floats at a time
        for (; j <= n - 8; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(&A[i * n + j]);
            __m256 x_vec = _mm256_loadu_ps(&x[j]);
            __m256 prod = _mm256_mul_ps(a_vec, x_vec);
            sum = _mm256_add_ps(sum, prod);
        }

        // Horizontal sum of 8 elements
        __m256 t = _mm256_hadd_ps(sum, sum);
        t = _mm256_hadd_ps(t, t);
        y[i] = _mm_cvtss_f32(_mm256_castps256_ps128(t));

        // Scalar remainder
        for (; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

void matmul_inner_alt(const float* A, const float* x, float* y, int m, int n) {
    for (int i = 0; i < m; i++) {
        __m256 sum = _mm256_setzero_ps();
        int j = 0;
        const float* row = &A[i * n];

        for (; j <= n - 8; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(&row[j]);
            __m256 x_vec = _mm256_loadu_ps(&x[j]);
            sum = _mm256_fmadd_ps(a_vec, x_vec, sum);
        }

        __m256 t = _mm256_hadd_ps(sum, sum);
        t = _mm256_hadd_ps(t, t);
        y[i] = _mm_cvtss_f32(_mm256_castps256_ps128(t));

        for (; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

void matmul_innervec_outerunrolled(const float* A, const float* x, float* y, int m, int n) {
    int rowend = m - m % 8;
    for (int row = 0; row < rowend; row += 8) {
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        __m256 s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps();
        __m256 s7 = _mm256_setzero_ps();
        __m256 s8 = _mm256_setzero_ps();

        int col = 0;
        for (; col <= n - 8; col += 8) {
            __m256 vx = _mm256_loadu_ps(&x[col]);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[row*n+col]), vx, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(row+1)*n+col]), vx, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(row+2)*n+col]), vx, s3);
            s4 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(row+3)*n+col]), vx, s4);
            s5 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(row+4)*n+col]), vx, s5);
            s6 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(row+5)*n+col]), vx, s6);
            s7 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(row+6)*n+col]), vx, s7);
            s8 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(row+7)*n+col]), vx, s8);
        }

        __m256 t1 = _mm256_hadd_ps(s1, s1); t1 = _mm256_hadd_ps(t1, t1);
        __m256 t2 = _mm256_hadd_ps(s2, s2); t2 = _mm256_hadd_ps(t2, t2);
        __m256 t3 = _mm256_hadd_ps(s3, s3); t3 = _mm256_hadd_ps(t3, t3);
        __m256 t4 = _mm256_hadd_ps(s4, s4); t4 = _mm256_hadd_ps(t4, t4);
        __m256 t5 = _mm256_hadd_ps(s5, s5); t5 = _mm256_hadd_ps(t5, t5);
        __m256 t6 = _mm256_hadd_ps(s6, s6); t6 = _mm256_hadd_ps(t6, t6);
        __m256 t7 = _mm256_hadd_ps(s7, s7); t7 = _mm256_hadd_ps(t7, t7);
        __m256 t8 = _mm256_hadd_ps(s8, s8); t8 = _mm256_hadd_ps(t8, t8);

        y[row] = _mm_cvtss_f32(_mm256_castps256_ps128(t1));
        y[row+1] = _mm_cvtss_f32(_mm256_castps256_ps128(t2));
        y[row+2] = _mm_cvtss_f32(_mm256_castps256_ps128(t2));
        y[row+3] = _mm_cvtss_f32(_mm256_castps256_ps128(t2));
        y[row+4] = _mm_cvtss_f32(_mm256_castps256_ps128(t2));
        y[row+5] = _mm_cvtss_f32(_mm256_castps256_ps128(t2));
        y[row+6] = _mm_cvtss_f32(_mm256_castps256_ps128(t2));
        y[row+7] = _mm_cvtss_f32(_mm256_castps256_ps128(t2));

        for (; col < n; col++) {
            y[row] += A[row*n+col] * x[col];
            y[row+1] += A[(row+1)*n+col] * x[col];
            y[row+2] += A[(row+2)*n+col] * x[col];
            y[row+3] += A[(row+3)*n+col] * x[col];
            y[row+4] += A[(row+4)*n+col] * x[col];
            y[row+5] += A[(row+5)*n+col] * x[col];
            y[row+6] += A[(row+6)*n+col] * x[col];
            y[row+7] += A[(row+7)*n+col] * x[col];            
        }
    }
    for (int row = rowend; row < m; row++) {
        __m256 sum = _mm256_setzero_ps();
        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(&A[row * n]);
            __m256 x_vec = _mm256_loadu_ps(&x[j]);
            sum = _mm256_fmadd_ps(a_vec, x_vec, sum);
        }
        __m256 t = _mm256_hadd_ps(sum, sum);
        t = _mm256_hadd_ps(t, t);
        y[row] += _mm_cvtss_f32(_mm256_castps256_ps128(t));
        for (; j < n; j++) { y[row] += A[row * n + j] * x[j]; }
    }
}

// maybe could use _mm256_maddubs_epi16 , but doesn't exist for floats
// _mm256_stream_ps to load by row?

// 2D vectorization (SIMD on both i and j)
void matmul_2d_vec(const float* A, const float* x, float* y, int m, int n) {
  // TODO: Implement 2D vectorization using AVX2  intrinsics
}
