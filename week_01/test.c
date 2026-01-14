#include "matmul.h"
#include <stdio.h>
#include <stdlib.h>

static void run_case(size_t ar, size_t ac, size_t br, size_t bc) {
    if (ac != br) {
        printf("SKIP incompatible: A=%zux%zu B=%zux%zu\n", ar, ac, br, bc);
        return;
    }

    Matrix A = mat_alloc(ar, ac);
    Matrix B = mat_alloc(br, bc);
    Matrix C1 = mat_alloc(ar, bc);
    Matrix C2 = mat_alloc(ar, bc);

    mat_fill_random(&A, 123);
    mat_fill_random(&B, 456);

    if (matmul_single(&A, &B, &C1) != 0) {
        printf("FAIL single dims A=%zux%zu B=%zux%zu\n", ar, ac, br, bc);
        exit(1);
    }

    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    for (size_t i = 0; i < sizeof(thread_counts)/sizeof(thread_counts[0]); i++) {
        int t = thread_counts[i];
        mat_fill(&C2, 0.0);
        int rc = matmul_pthreads(&A, &B, &C2, t);
        if (rc != 0) {
            printf("FAIL pthread rc=%d threads=%d A=%zux%zu B=%zux%zu\n",
                   rc, t, ar, ac, br, bc);
            exit(1);
        }
        if (!mat_equal_eps(&C1, &C2, 1e-9)) {
            printf("MISMATCH threads=%d A=%zux%zu B=%zux%zu\n", t, ar, ac, br, bc);
            exit(1);
        }
    }

    printf("PASS A=%zux%zu B=%zux%zu\n", ar, ac, br, bc);

    mat_free(&A); mat_free(&B); mat_free(&C1); mat_free(&C2);
}

int main(void) {
    // required examples
    run_case(1,1,1,1);
    run_case(1,1,1,5);
    run_case(2,1,1,3);
    run_case(2,2,2,2);

    // more corner / variety
    run_case(1,7,7,1);
    run_case(3,5,5,4);
    run_case(5,3,3,9);
    run_case(8,8,8,8);
    run_case(16,1,1,16);
    run_case(31,17,17,29);
    run_case(64,64,64,64);

    // non-square but larger
    run_case(128,64,64,256);
    run_case(256,128,128,64);

    printf("All tests passed.\n");
    return 0;
}
