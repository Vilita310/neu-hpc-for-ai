\
#include "flashattn.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static void usage() {
    printf("Usage: flashattn_test [--N N] [--d d] [--Br Br] [--Bc Bc] [--seed s]\n");
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

static float rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double da = (double)a[i];
        double db = (double)b[i];
        double diff = da - db;
        num += diff * diff;
        den += db * db;
    }
    return (float)std::sqrt(num / (den + 1e-12));
}

int main(int argc, char** argv) {
    int N = 256, d = 64, Br = 32, Bc = 32;
    int seed = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_i = [&]() -> int { if (i + 1 >= argc) { usage(); std::exit(1); } return ++i; };
        if (arg == "--N") N = std::atoi(argv[next_i()]);
        else if (arg == "--d") d = std::atoi(argv[next_i()]);
        else if (arg == "--Br") Br = std::atoi(argv[next_i()]);
        else if (arg == "--Bc") Bc = std::atoi(argv[next_i()]);
        else if (arg == "--seed") seed = std::atoi(argv[next_i()]);
        else { usage(); return 1; }
    }

    printf("Config: N=%d d=%d Br=%d Bc=%d seed=%d\n", N, d, Br, Bc, seed);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t bytes = (size_t)N * d;
    std::vector<float> Q(bytes), K(bytes), V(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        Q[i] = dist(rng);
        K[i] = dist(rng);
        V[i] = dist(rng);
    }

    std::vector<float> O_naive(bytes), O_cpu(bytes), O_gpu(bytes);

    printf("Running CPU naive...\n");
    naive_attention_cpu(Q.data(), K.data(), V.data(), O_naive.data(), N, d);

    printf("Running CPU FlashAttention Alg1...\n");
    flash_attention_alg1_cpu(Q.data(), K.data(), V.data(), O_cpu.data(), N, d, Br, Bc);

    float cpu_max = max_abs_diff(O_cpu, O_naive);
    float cpu_rel = rel_l2(O_cpu, O_naive);
    printf("[CPU Flash vs Naive] max_abs=%.6g  rel_l2=%.6g\n", cpu_max, cpu_rel);

    printf("Running CUDA FlashAttention Alg1...\n");
    int rc = flash_attention_alg1_cuda(Q.data(), K.data(), V.data(), O_gpu.data(), N, d, Br, Bc);
    if (rc != 0) {
        printf("CUDA implementation failed with error code %d\n", rc);
        return 2;
    }

    float gpu_max = max_abs_diff(O_gpu, O_naive);
    float gpu_rel = rel_l2(O_gpu, O_naive);
    printf("[CUDA Flash vs Naive] max_abs=%.6g  rel_l2=%.6g\n", gpu_max, gpu_rel);

    // Basic pass/fail thresholds
    const float max_tol = 5e-3f;
    const float rel_tol = 5e-3f;

    bool pass = (cpu_max < max_tol && cpu_rel < rel_tol && gpu_max < max_tol && gpu_rel < rel_tol);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL (tolerances too strict or bug present)");
    return pass ? 0 : 3;
}
