#define _POSIX_C_SOURCE 200809L
#include "matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* get current time in seconds */
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double time_single(const Matrix *A, const Matrix *B, Matrix *C, int iters) {
    double best = 1e100;
    for (int i = 0; i < iters; i++) {
        mat_fill(C, 0.0);
        double t0 = now_s();
        matmul_single(A, B, C);
        double t1 = now_s();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }
    return best;
}

static double time_pthreads(const Matrix *A, const Matrix *B, Matrix *C,
                            int threads, int iters) {
    double best = 1e100;
    for (int i = 0; i < iters; i++) {
        mat_fill(C, 0.0);
        double t0 = now_s();
        matmul_pthreads(A, B, C, threads);
        double t1 = now_s();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char **argv) {
    size_t N = 1024;   // default matrix size
    int iters = 3;     // default iterations

    if (argc >= 2) N = (size_t)atoi(argv[1]);
    if (argc >= 3) iters = atoi(argv[2]);

    if (N == 0 || iters <= 0) {
        fprintf(stderr, "Usage: %s [matrix_size] [iters]\n", argv[0]);
        return 1;
    }

    printf("Benchmark matmul: A=%zux%zu B=%zux%zu (iters=%d)\n",
           N, N, N, N, iters);

    Matrix A = mat_alloc(N, N);
    Matrix B = mat_alloc(N, N);
    Matrix C = mat_alloc(N, N);

    mat_fill_random(&A, 123);
    mat_fill_random(&B, 456);

    /* ---------- CSV OUTPUT ---------- */
    FILE *csv = fopen("results.csv", "w");
    if (!csv) {
        perror("fopen results.csv");
        return 1;
    }
    fprintf(csv, "threads,time_seconds,speedup\n");

    /* ---------- BASELINE ---------- */
    double t1 = time_single(&A, &B, &C, iters);
    printf("single-thread: %.6f s (baseline)\n", t1);
    fprintf(csv, "1,%.6f,1.00\n", t1);

    /* ---------- MULTI-THREAD ---------- */
    int thread_counts[] = {1, 4, 16, 32, 64, 128};
    size_t num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (size_t i = 0; i < num_tests; i++) {
        int th = thread_counts[i];
        double tp = time_pthreads(&A, &B, &C, th, iters);
        double speedup = t1 / tp;

        printf("threads=%3d  time=%.6f s  speedup=%.2fx\n",
               th, tp, speedup);

        fprintf(csv, "%d,%.6f,%.2f\n", th, tp, speedup);
    }

    fclose(csv);

    mat_free(&A);
    mat_free(&B);
    mat_free(&C);

    printf("Results written to results.csv\n");
    return 0;
}
