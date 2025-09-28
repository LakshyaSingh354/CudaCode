#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <random>

// ------------------- Weights: row-wise absmax (symmetric) -------------------
void quantize_weights_rowwise_cpu(
    const std::vector<float>& W,
    std::vector<int8_t>& Wq,
    std::vector<float>& rowScales,
    int M, int K
) {
    Wq.resize(M * K);
    rowScales.resize(M);

    for (int m = 0; m < M; m++) {
        float maxv = 0.0f;
        for (int k = 0; k < K; k++) {
            maxv = std::max(maxv, std::fabs(W[m*K + k]));
        }
        float scale = (maxv == 0.0f) ? 1.0f : maxv / 127.0f;
        rowScales[m] = scale;

        for (int k = 0; k < K; k++) {
            float val = W[m*K + k] / scale;
            int q = static_cast<int>(std::round(val));
            q = std::max(-127, std::min(127, q));
            Wq[m*K + k] = static_cast<int8_t>(q);
        }
    }
}

// ------------------- Activations: col-wise min/max (asymmetric) -------------------
void quantize_activations_colwise_cpu(
    const std::vector<float>& X,
    std::vector<uint8_t>& Xq,
    std::vector<float>& colScales,
    std::vector<int32_t>& colZPs,
    int N, int K
) {
    Xq.resize(N * K);
    colScales.resize(K);
    colZPs.resize(K);

    for (int n = 0; n < K; n++) {
        float minv = 1e30f;
        float maxv = -1e30f;
        for (int m = 0; m < N; m++) {
            float val = X[m*K + n];
            minv = std::min(minv, val);
            maxv = std::max(maxv, val);
        }
        float scale = (maxv - minv) / 255.0f;
        if (scale == 0.0f) scale = 1.0f;
        int zp = static_cast<int>(std::round(-minv / scale));
        zp = std::max(0, std::min(255, zp));

        colScales[n] = scale;
        colZPs[n] = zp;

        for (int m = 0; m < N; m++) {
            float val = X[m*K + n] / scale + zp;  // quantize
            int q = static_cast<int>(std::round(val));
            q = std::max(0, std::min(255, q));
            Xq[m*K + n] = static_cast<uint8_t>(q);
        }
    }
}

// ------------------- GEMM with dequant -------------------
void qgemm_cpu(
    const std::vector<int8_t>& Wq, const std::vector<float>& rowScales,
    const std::vector<uint8_t>& Xq, const std::vector<float>& colScales,
    const std::vector<int32_t>& colZPs,
    std::vector<float>& Yq,
    int M, int K, int N
) {
    Yq.assign(M * N, 0.0f);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int32_t w = Wq[m*K + k];
                int32_t x = static_cast<int32_t>(Xq[n*K + k]) - colZPs[k]; // careful: row-major [N,K]
                acc += w * x;
            }
            Yq[m*N + n] = static_cast<float>(acc) * (rowScales[m] * colScales[n]);
        }
    }
}

// ------------------- Helpers -------------------
float mse(const std::vector<float>& A, const std::vector<float>& B) {
    float err = 0.0f;
    for (size_t i = 0; i < A.size(); i++) {
        float d = A[i] - B[i];
        err += d * d;
    }
    return err / A.size();
}

// ------------------- Main -------------------
int main() {
    int M = 512;   // out dim
    int K = 1024;  // inner dim
    int N = 4096;  // batch size

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> W(M*K), X(N*K);

    for (auto& w : W) w = dist(rng);
    for (auto& x : X) x = dist(rng);

    // Float reference GEMM: Y = X @ W^T  -> [N,M]
    std::vector<float> Yref(N*M, 0.0f);
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += X[n*K + k] * W[m*K + k];
            }
            Yref[n*M + m] = acc;
        }
    }

    // Quantize weights
    std::vector<int8_t> Wq;
    std::vector<float> rowScales;
    quantize_weights_rowwise_cpu(W, Wq, rowScales, M, K);

    // Dequant weights and check error
    std::vector<float> Wdeq(M*K);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            Wdeq[m*K + k] = static_cast<float>(Wq[m*K + k]) * rowScales[m];
        }
    }
    std::cout << "MSE W: " << mse(W, Wdeq) << "\n";

    // Quantize activations
    std::vector<uint8_t> Xq;
    std::vector<float> colScales;
    std::vector<int32_t> colZPs;
    quantize_activations_colwise_cpu(X, Xq, colScales, colZPs, N, K);

    // Dequant activations and check error
    std::vector<float> Xdeq(N*K);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            Xdeq[n*K + k] = (static_cast<int>(Xq[n*K + k]) - colZPs[k]) * colScales[k];
        }
    }
    std::cout << "MSE X: " << mse(X, Xdeq) << "\n";

    // Quantized GEMM
    std::vector<float> Yq;
    qgemm_cpu(Wq, rowScales, Xq, colScales, colZPs, Yq, M, K, N);

    std::cout << "MSE GEMM: " << mse(Yref, Yq) << "\n";

    return 0;
}