#include <cuda_runtime.h>

/*
 * ThunderKittens integration note:
 * - If TK headers/toolchain are available, define TK_AVAILABLE and compile this
 *   file with kittens.cuh in include path.
 * - Otherwise the fallback kernel keeps the extension runnable.
 */

#ifdef TK_AVAILABLE
#include "kittens.cuh"
using namespace kittens;

static constexpr int BLOCK_SIZE = 32;
static constexpr int NUM_WORKERS = 1;
static constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;

struct tk_globals {
    using sub_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using tile_gl = gl<bf16, 1, 1, -1, -1, sub_tile>;
    tile_gl A;
    tile_gl B;
    tile_gl C;
    int N;
};

__global__ void tk_wmma_tma_kernel(const __grid_constant__ tk_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_bf<BLOCK_SIZE, BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();

    rt_bf<BLOCK_SIZE, BLOCK_SIZE> A_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> B_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::col> B_reg_col;
    rt_fl<BLOCK_SIZE, BLOCK_SIZE> C_accum;

    const int row = blockIdx.y;
    const int col = blockIdx.x;
    const int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    warp::zero(C_accum);
    for (int tile = 0; tile < num_tiles; ++tile) {
        warp::load(As, g.A, {0, 0, row, tile});
        warp::load(Bs, g.B, {0, 0, tile, col});
        __syncthreads();

        warp::load(A_reg, As);
        warp::load(B_reg, Bs);
        warp::swap_layout(B_reg_col, B_reg);
        warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);
        __syncthreads();
    }
    warp::store(g.C, C_accum, {0, 0, row, col});
}
#endif

extern "C" __global__ void tk_moe_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
