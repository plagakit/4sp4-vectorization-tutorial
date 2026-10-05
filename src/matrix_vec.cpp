#include <immintrin.h>
#include "matrix_vec.h"

#include <vector>

// inline __attribute__((always_inline)) float horzsum256(__m256 s) {
//     // https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
//     auto s2 = _mm256_permute2f128_ps(s , s , 1);
//     s = _mm256_add_ps(s, s2);
//     s = _mm256_hadd_ps(s, s);
//     s = _mm256_hadd_ps(s, s);
//     return _mm_cvtss_f32(_mm256_castps256_ps128(s));
// }

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
        // can't vectorize bc the += is sequential?
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
        auto s2 = _mm256_permute2f128_ps(sum, sum, 1);
        sum = _mm256_add_ps(sum, s2);
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        y[i] = _mm_cvtss_f32(_mm256_castps256_ps128(sum));

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

        auto s2 = _mm256_permute2f128_ps(sum, sum, 1);
        sum = _mm256_add_ps(sum, s2);
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        y[i] = _mm_cvtss_f32(_mm256_castps256_ps128(sum));

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

        auto t1 = _mm256_permute2f128_ps(s1, s1, 1); s1 = _mm256_add_ps(s1, t1); s1 = _mm256_hadd_ps(s1, s1); s1 = _mm256_hadd_ps(s1, s1);
        auto t2 = _mm256_permute2f128_ps(s2, s2, 1); s2 = _mm256_add_ps(s2, t2); s2 = _mm256_hadd_ps(s2, s2); s2 = _mm256_hadd_ps(s2, s2);
        auto t3 = _mm256_permute2f128_ps(s3, s3, 1); s3 = _mm256_add_ps(s3, t3); s3 = _mm256_hadd_ps(s3, s3); s3 = _mm256_hadd_ps(s3, s3);
        auto t4 = _mm256_permute2f128_ps(s4, s4, 1); s4 = _mm256_add_ps(s4, t4); s4 = _mm256_hadd_ps(s4, s4); s4 = _mm256_hadd_ps(s4, s4);
        auto t5 = _mm256_permute2f128_ps(s5, s5, 1); s5 = _mm256_add_ps(s5, t5); s5 = _mm256_hadd_ps(s5, s5); s5 = _mm256_hadd_ps(s5, s5);
        auto t6 = _mm256_permute2f128_ps(s6, s6, 1); s6 = _mm256_add_ps(s6, t6); s6 = _mm256_hadd_ps(s6, s6); s6 = _mm256_hadd_ps(s6, s6);
        auto t7 = _mm256_permute2f128_ps(s7, s7, 1); s7 = _mm256_add_ps(s7, t7); s7 = _mm256_hadd_ps(s7, s7); s7 = _mm256_hadd_ps(s7, s7);
        auto t8 = _mm256_permute2f128_ps(s8, s8, 1); s8 = _mm256_add_ps(s8, t8); s8 = _mm256_hadd_ps(s8, s8); s8 = _mm256_hadd_ps(s8, s8);

        y[row] = _mm_cvtss_f32(_mm256_castps256_ps128(s1));
        y[row+1] = _mm_cvtss_f32(_mm256_castps256_ps128(s2));
        y[row+2] = _mm_cvtss_f32(_mm256_castps256_ps128(s3));
        y[row+3] = _mm_cvtss_f32(_mm256_castps256_ps128(s4));
        y[row+4] = _mm_cvtss_f32(_mm256_castps256_ps128(s5));
        y[row+5] = _mm_cvtss_f32(_mm256_castps256_ps128(s6));
        y[row+6] = _mm_cvtss_f32(_mm256_castps256_ps128(s7));
        y[row+7] = _mm_cvtss_f32(_mm256_castps256_ps128(s8));

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
        auto s2 = _mm256_permute2f128_ps(sum, sum, 1);
        sum = _mm256_add_ps(sum, s2);
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        y[row] += _mm_cvtss_f32(_mm256_castps256_ps128(sum));
        for (; j < n; j++) { y[row] += A[row * n + j] * x[j]; }
    }
}


void matmul_vec_outer_vec(const float* A, const float* x, float* y, int m, int n) {
    // do 8 rows at a time, but each column iteratively
    int row = 0;
    for (; row < m - m % 8; row += 8) {   
        __m256 sum = _mm256_setzero_ps();
        __m256 vx = _mm256_loadu_ps(&x[row]);
        __m256i gather_offsets = _mm256_set_epi32(
            row*n,
            (row+1)*n,
            (row+2)*n,
            (row+3)*n,
            (row+4)*n,
            (row+5)*n,
            (row+6)*n,
            (row+7)*n
        );

        for (int col = 0; col < n; col++) {

            __m256i col_idx = _mm256_set1_epi32(col);
            __m256i idxs = _mm256_add_epi32(col_idx, gather_offsets);

            __m256 va = _mm256_i32gather_ps(A, idxs, sizeof(float));
            sum = _mm256_fmadd_ps(va, vx, sum);
        }

        _mm256_storeu_ps(&y[row], sum);
    }

    // clean up the rest
    for (int i = row; i < m; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

// 2D vectorization (SIMD on both i and j)
void matmul_2d_vec(const float* A, const float* x, float* y, int m, int n) {

}



// for j := 0-7
//     i := j * 32 (0, 32, 64, 96, 128, ...)
//     m := id_t

//     addr := A + 

//     dst[i+31:i] := MEM[base_addr + SignExtend(vindex[i+31:i])*scale]

// dst[i+31:i] := MEM[base_addr + SignExtend(vindex[i+31:i])*scale]




// _mm256_i32gather_ps(A[i], idx, 8);