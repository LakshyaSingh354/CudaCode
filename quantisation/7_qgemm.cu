#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>

// ------------------- Kernels -------------------

__global__ void absmax_rowwise_kernel(const float* __restrict__ A, float* rowScales, int M, int K) {
    int m = blockIdx.x;
    if (m < M) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        float local_max = 0.f;
        for (int k = tid; k < K; k += blockDim.x) {
            float val = fabsf(A[m * K + k]);
            local_max = fmaxf(local_max, val);
        }
        sdata[tid] = local_max;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            rowScales[m] = fmaxf(sdata[0] / 127.f, 1e-8f);
        }
    }
}

__global__ void colwise_minmax(const float* __restrict__ A, float* colMins, float* colMaxs, int M, int N) {
    int n = blockIdx.x;
    if (n < N) {
        extern __shared__ float sdata[];
        float* smins = sdata;
        float* smaxs = sdata + blockDim.x;

        int tid = threadIdx.x;
        float local_min = 1e30f;
        float local_max = -1e30f;
        for (int i = tid; i < M; i += blockDim.x) {
            float val = A[i * N + n];
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
        smins[tid] = local_min;
        smaxs[tid] = local_max;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smins[tid] = fminf(smins[tid], smins[tid + s]);
                smaxs[tid] = fmaxf(smaxs[tid], smaxs[tid + s]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            colMins[n] = smins[0];
            colMaxs[n] = smaxs[0];
        }
    }
}

__global__ void quantize_weights_rowwise(const float* __restrict__ A, int8_t* Aq,
                                         const float* __restrict__ rowScales, int M, int K) {
    int m = blockIdx.x;
    int k = threadIdx.x;
    if (m < M && k < K) {
        float scale = rowScales[m];
        float val = A[m * K + k] / scale;
        val = fmaxf(fminf(val, 127.f), -127.f);
        Aq[m * K + k] = static_cast<int8_t>(rintf(val));
    }
}

__global__ void quantize_activations_colwise(const float* __restrict__ X, uint8_t* Xq,
                                             const float* __restrict__ colMins,
                                             const float* __restrict__ colMaxs,
                                             float* colScales, int* colZPs,
                                             int M, int N) {
    int n = blockIdx.x;
    int m = threadIdx.x;
    if (n < N && m < M) {
        float minv = colMins[n];
        float maxv = colMaxs[n];
        float scale = fmaxf((maxv - minv) / 255.f, 1e-8f);
        int zp = static_cast<int>(rintf(-minv / scale));
        colScales[n] = scale;
        colZPs[n] = zp;

        float val = X[m * N + n] / scale + zp;
        val = fmaxf(fminf(val, 255.f), 0.f);
        Xq[m * N + n] = static_cast<uint8_t>(rintf(val));
    }
}

__global__ void qgemm_kernel(const int8_t* __restrict__ Wq, const float* __restrict__ Sw,
                             const uint8_t* __restrict__ Xq, const float* __restrict__ Sx,
                             const int* __restrict__ Zx, float* Y, int M, int K, int N) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && n < N) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++) {
            int32_t w = static_cast<int32_t>(Wq[m * K + k]);
            int32_t x = static_cast<int32_t>(Xq[k * N + n]) - Zx[n];
            acc += w * x;
        }
        float result = acc * (Sw[m] * Sx[n]);
        Y[m * N + n] = result;
    }
}

// ------------------- Host-side reference GEMM -------------------

void reference_fp32_gemm(const std::vector<float>& W, const std::vector<float>& X,
                         std::vector<float>& Y, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) {
                acc += W[m * K + k] * X[k * N + n];
            }
            Y[m * N + n] = acc;
        }
    }
}

void compute_mse(const std::vector<float>& Y_ref, const std::vector<float>& Y_q,
                 int M, int N) {
    double mse = 0.0;
    double max_abs_err = 0.0;
    int size = M * N;
    for (int i = 0; i < size; i++) {
        double diff = Y_ref[i] - Y_q[i];
        mse += diff * diff;
        max_abs_err = std::max(max_abs_err, fabs(diff));
    }
    mse /= size;
    std::cout << "MSE: " << mse << ", MaxAbsErr: " << max_abs_err << std::endl;
}

// ------------------- Main -------------------

int main() {
    int M = 4096, K = 4096, N = 4096;

    std::vector<float> hW(M * K);
    std::vector<float> hX(K * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v : hW) v = dist(rng);
    for (auto& v : hX) v = dist(rng);

    // device buffers
    float *dW, *dX;
    cudaMalloc(&dW, M * K * sizeof(float));
    cudaMalloc(&dX, K * N * sizeof(float));
    cudaMemcpy(dW, hW.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    int8_t* dWq; uint8_t* dXq;
    cudaMalloc(&dWq, M * K * sizeof(int8_t));
    cudaMalloc(&dXq, K * N * sizeof(uint8_t));

    float *dW_scales, *dX_colMins, *dX_colMaxs, *dX_scales;
    int* dX_zps;
    cudaMalloc(&dW_scales, M * sizeof(float));
    cudaMalloc(&dX_colMins, N * sizeof(float));
    cudaMalloc(&dX_colMaxs, N * sizeof(float));
    cudaMalloc(&dX_scales, N * sizeof(float));
    cudaMalloc(&dX_zps, N * sizeof(int));

    absmax_rowwise_kernel<<<M, 256, 256 * sizeof(float)>>>(dW, dW_scales, M, K);
    cudaDeviceSynchronize();

    quantize_weights_rowwise<<<M, K>>>(dW, dWq, dW_scales, M, K);
    cudaDeviceSynchronize();

    colwise_minmax<<<N, 256, 2 * 256 * sizeof(float)>>>(dX, dX_colMins, dX_colMaxs, K, N);
    cudaDeviceSynchronize();

    quantize_activations_colwise<<<N, K>>>(dX, dXq, dX_colMins, dX_colMaxs,
                                           dX_scales, dX_zps, K, N);
    cudaDeviceSynchronize();

    float* dY;
    cudaMalloc(&dY, M * N * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    qgemm_kernel<<<grid, block>>>(dWq, dW_scales, dXq, dX_scales, dX_zps, dY, M, K, N);
    cudaDeviceSynchronize();

    std::vector<float> hY(M * N);
    cudaMemcpy(hY.data(), dY, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // ------------------- Reference + Error -------------------
    std::vector<float> hY_ref(M * N);
    reference_fp32_gemm(hW, hX, hY_ref, M, K, N);

    // std::cout << "Quantized GEMM result:" << std::endl;
    // for (int m = 0; m < M; m++) {
    //     for (int n = 0; n < N; n++) {
    //         std::cout << hY[m * N + n] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Reference FP32 GEMM result:" << std::endl;
    // for (int m = 0; m < M; m++) {
    //     for (int n = 0; n < N; n++) {
    //         std::cout << hY_ref[m * N + n] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    compute_mse(hY_ref, hY, M, N);

    cudaFree(dW); cudaFree(dX); cudaFree(dWq); cudaFree(dXq);
    cudaFree(dW_scales); cudaFree(dX_colMins); cudaFree(dX_colMaxs);
    cudaFree(dX_scales); cudaFree(dX_zps); cudaFree(dY);

    return 0;
}