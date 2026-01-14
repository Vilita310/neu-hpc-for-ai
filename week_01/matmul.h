#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

typedef struct {
    size_t rows;
    size_t cols;
    double *data;  // row-major: data[i*cols + j]
} Matrix;

// basic utils
Matrix mat_alloc(size_t rows, size_t cols);
void   mat_free(Matrix *m);
double mat_get(const Matrix *m, size_t i, size_t j);
void   mat_set(Matrix *m, size_t i, size_t j, double v);
void   mat_fill(Matrix *m, double v);
void   mat_fill_random(Matrix *m, unsigned int seed);
int    mat_equal_eps(const Matrix *a, const Matrix *b, double eps);

// reference + implementations
int matmul_single(const Matrix *A, const Matrix *B, Matrix *C);
int matmul_pthreads(const Matrix *A, const Matrix *B, Matrix *C, int num_threads);

#endif
