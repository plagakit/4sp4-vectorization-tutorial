#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

void matmult_base(const float* A, const float* B, float* C, int n);
void matmult_ikj(const float* A, const float* B, float* C, int n);
void matmult_tiled(const float* A, const float* B, float* C, int n);
void matmult_tiled_ikj(const float* A, const float* B, float* C, int n);

#endif

