#define _POSIX_C_SOURCE 200809L
#include "matmul.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

Matrix mat_alloc(size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double*)calloc(rows * cols, sizeof(double));
    if (!m.data) {
        fprintf(stderr, "mat_alloc failed: %zu x %zu\n", rows, cols);
        m.rows = m.cols = 0;
    }
    return m;
}

void mat_free(Matrix *m) {
    if (!m) return;
    free(m->data);
    m->data = NULL;
    m->rows = m->cols = 0;
}

double mat_get(const Matrix *m, size_t i, size_t j) {
    return m->data[i * m->cols + j];
}

void mat_set(Matrix *m, size_t i, size_t j, double v) {
    m->data[i * m->cols + j] = v;
}

void mat_fill(Matrix *m, double v) {
    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) m->data[i] = v;
}

void mat_fill_random(Matrix *m, unsigned int seed) {
    // deterministic random
    srand(seed);
    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) {
        // [-1, 1]
        double r = (double)rand() / (double)RAND_MAX;
        m->data[i] = 2.0 * r - 1.0;
    }
}

int mat_equal_eps(const Matrix *a, const Matrix *b, double eps) {
    if (a->rows != b->rows || a->cols != b->cols) return 0;
    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++) {
        double diff = fabs(a->data[i] - b->data[i]);
        if (diff > eps) return 0;
    }
    return 1;
}

static int check_dims(const Matrix *A, const Matrix *B, Matrix *C) {
    if (!A || !B || !C) return 0;
    if (!A->data || !B->data || !C->data) return 0;
    if (A->cols != B->rows) return 0;
    if (C->rows != A->rows || C->cols != B->cols) return 0;
    return 1;
}

// -------------------- Single-thread --------------------
int matmul_single(const Matrix *A, const Matrix *B, Matrix *C) {
    if (!check_dims(A, B, C)) return -1;

    // Ensure C starts at 0
    mat_fill(C, 0.0);

    size_t M = A->rows, K = A->cols, N = B->cols;

    // i x k times k x j
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            double a = A->data[i*K + k];
            const double *brow = &B->data[k*N];
            double *crow = &C->data[i*N];
            for (size_t j = 0; j < N; j++) {
                crow[j] += a * brow[j];
            }
        }
    }
    return 0;
}

// -------------------- Pthreads --------------------
typedef struct {
    const Matrix *A;
    const Matrix *B;
    Matrix *C;
    size_t row_start;
    size_t row_end; // [start, end)
} WorkerArgs;

static void* worker_mul(void *arg) {
    WorkerArgs *w = (WorkerArgs*)arg;
    const Matrix *A = w->A;
    const Matrix *B = w->B;
    Matrix *C = w->C;

    size_t M = A->rows, K = A->cols, N = B->cols;
    (void)M;

    for (size_t i = w->row_start; i < w->row_end; i++) {
        double *crow = &C->data[i*N];
        for (size_t k = 0; k < K; k++) {
            double a = A->data[i*K + k];
            const double *brow = &B->data[k*N];
            for (size_t j = 0; j < N; j++) {
                crow[j] += a * brow[j];
            }
        }
    }
    return NULL;
}

int matmul_pthreads(const Matrix *A, const Matrix *B, Matrix *C, int num_threads) {
    if (!check_dims(A, B, C)) return -1;
    if (num_threads <= 0) return -2;

    mat_fill(C, 0.0);

    size_t M = A->rows;
    int t = num_threads;
    if ((size_t)t > M) t = (int)M; // more threads than rows is pointless

    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)t);
    WorkerArgs *args   = (WorkerArgs*)malloc(sizeof(WorkerArgs) * (size_t)t);
    if (!threads || !args) {
        free(threads); free(args);
        return -3;
    }

    size_t base = M / (size_t)t;
    size_t rem  = M % (size_t)t;

    size_t cur = 0;
    for (int i = 0; i < t; i++) {
        size_t extra = (size_t)(i < (int)rem ? 1 : 0);
        size_t start = cur;
        size_t end   = cur + base + extra;
        cur = end;

        args[i] = (WorkerArgs){
            .A = A, .B = B, .C = C,
            .row_start = start,
            .row_end = end
        };

        int rc = pthread_create(&threads[i], NULL, worker_mul, &args[i]);
        if (rc != 0) {
            // join created threads
            for (int j = 0; j < i; j++) pthread_join(threads[j], NULL);
            free(threads); free(args);
            return -4;
        }
    }

    for (int i = 0; i < t; i++) pthread_join(threads[i], NULL);

    free(threads);
    free(args);
    return 0;
}
