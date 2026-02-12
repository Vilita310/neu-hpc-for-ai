\
#include "flashattn.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

// Row-major indexing helpers
static inline const float* row_ptr(const float* A, int row, int d) { return A + (size_t)row * d; }
static inline float* row_ptr(float* A, int row, int d) { return A + (size_t)row * d; }

void naive_attention_cpu(const float* Q, const float* K, const float* V,
                         float* O, int N, int d) {
    const float scale = 1.0f / std::sqrt((float)d);

    std::vector<float> logits((size_t)N);
    std::vector<float> probs((size_t)N);

    for (int i = 0; i < N; ++i) {
        const float* qi = row_ptr(Q, i, d);

        // logits_j = dot(qi, Kj) * scale
        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) {
            const float* kj = row_ptr(K, j, d);
            float dot = 0.0f;
            for (int x = 0; x < d; ++x) dot += qi[x] * kj[x];
            float s = dot * scale;
            logits[j] = s;
            m = std::max(m, s);
        }

        // softmax
        float l = 0.0f;
        for (int j = 0; j < N; ++j) {
            float p = std::exp(logits[j] - m);
            probs[j] = p;
            l += p;
        }
        float inv_l = 1.0f / l;

        // output
        float* oi = row_ptr(O, i, d);
        for (int x = 0; x < d; ++x) oi[x] = 0.0f;
        for (int j = 0; j < N; ++j) {
            const float* vj = row_ptr(V, j, d);
            float p = probs[j] * inv_l;
            for (int x = 0; x < d; ++x) oi[x] += p * vj[x];
        }
    }
}

void flash_attention_alg1_cpu(const float* Q, const float* K, const float* V,
                              float* O, int N, int d, int Br, int Bc) {
    const float scale = 1.0f / std::sqrt((float)d);

    // Work per Q tile
    for (int qi0 = 0; qi0 < N; qi0 += Br) {
        int br = std::min(Br, N - qi0);

        std::vector<float> mi((size_t)br, -std::numeric_limits<float>::infinity());
        std::vector<float> li((size_t)br, 0.0f);
        std::vector<float> Oacc((size_t)br * d, 0.0f);

        // Loop over K/V tiles
        std::vector<float> s((size_t)Bc);
        std::vector<float> p((size_t)Bc);

        for (int kj0 = 0; kj0 < N; kj0 += Bc) {
            int bc = std::min(Bc, N - kj0);

            for (int r = 0; r < br; ++r) {
                const float* q = row_ptr(Q, qi0 + r, d);

                // (1) logits and block max
                float mij = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < bc; ++c) {
                    const float* k = row_ptr(K, kj0 + c, d);
                    float dot = 0.0f;
                    for (int x = 0; x < d; ++x) dot += q[x] * k[x];
                    float val = dot * scale;
                    s[c] = val;
                    mij = std::max(mij, val);
                }

                // (2) pij and block sum
                float lij = 0.0f;
                for (int c = 0; c < bc; ++c) {
                    float pc = std::exp(s[c] - mij);
                    p[c] = pc;
                    lij += pc;
                }

                // (3) online update
                float mi_old = mi[(size_t)r];
                float li_old = li[(size_t)r];

                float mi_new = std::max(mi_old, mij);
                float alpha = std::exp(mi_old - mi_new);
                float beta  = std::exp(mij    - mi_new);

                float li_new = alpha * li_old + beta * lij;

                // PV = sum_c p[c] * V[kj0+c]
                std::vector<float> PV((size_t)d, 0.0f);
                for (int c = 0; c < bc; ++c) {
                    const float* v = row_ptr(V, kj0 + c, d);
                    float pc = p[c];
                    for (int x = 0; x < d; ++x) PV[(size_t)x] += pc * v[x];
                }

                float* oacc = &Oacc[(size_t)r * d];
                for (int x = 0; x < d; ++x) {
                    oacc[x] = alpha * oacc[x] + beta * PV[(size_t)x];
                }

                mi[(size_t)r] = mi_new;
                li[(size_t)r] = li_new;
            }
        }

        // Final normalize
        for (int r = 0; r < br; ++r) {
            float inv = 1.0f / li[(size_t)r];
            float* out = row_ptr(O, qi0 + r, d);
            const float* oacc = &Oacc[(size_t)r * d];
            for (int x = 0; x < d; ++x) out[x] = oacc[x] * inv;
        }
    }
}
