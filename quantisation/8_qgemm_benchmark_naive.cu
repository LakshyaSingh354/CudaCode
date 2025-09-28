#include <cublas_v2.h>
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
    int M = 512, K = 1024, N = 4096;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // BENCHMARK: Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float qgemm_time = 0.0f;
    float cublas_time = 0.0f;

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

    // ------------------- Custom Quantized GEMM Execution -------------------
    std::cout << "Running Quantized GEMM..." << std::endl;

    int warmup_iters = 3;
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

    float* dY;
    cudaMalloc(&dY, M * N * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Warmup (not timed)
    for (int i = 0; i < warmup_iters; ++i) {
        absmax_rowwise_kernel<<<M, 256, 256 * sizeof(float)>>>(dW, dW_scales, M, K);
        quantize_weights_rowwise<<<dim3(M, (K + 255) / 256), 256>>>(dW, dWq, dW_scales, M, K);
        colwise_minmax<<<N, 256, 2 * 256 * sizeof(float)>>>(dX, dX_colMins, dX_colMaxs, K, N);
        quantize_activations_colwise<<<dim3(N, (K + 255) / 256), 256>>>(dX, dXq, dX_colMins, dX_colMaxs, dX_scales, dX_zps, K, N);
        qgemm_kernel<<<grid, block>>>(dWq, dW_scales, dXq, dX_scales, dX_zps, dY, M, K, N);
    }
    cudaDeviceSynchronize();

    // Timed run
    absmax_rowwise_kernel<<<M, 256, 256 * sizeof(float)>>>(dW, dW_scales, M, K);
    quantize_weights_rowwise<<<dim3(M, (K + 255) / 256), 256>>>(dW, dWq, dW_scales, M, K);
    colwise_minmax<<<N, 256, 2 * 256 * sizeof(float)>>>(dX, dX_colMins, dX_colMaxs, K, N);
    quantize_activations_colwise<<<dim3(N, (K + 255) / 256), 256>>>(dX, dXq, dX_colMins, dX_colMaxs, dX_scales, dX_zps, K, N);
    cudaEventRecord(start);
    qgemm_kernel<<<grid, block>>>(dWq, dW_scales, dXq, dX_scales, dX_zps, dY, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&qgemm_time, start, stop);

    std::vector<float> hY(M * N);
    cudaMemcpy(hY.data(), dY, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // ------------------- cuBLAS FP32 GEMM Execution -------------------
    std::cout << "Running cuBLAS FP32 SGEMM..." << std::endl;

    float* dY_cublas;
    cudaMalloc(&dY_cublas, M * N * sizeof(float));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warmup (not timed)
    for (int i = 0; i < warmup_iters; ++i) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    dX, N,
                    dW, K,
                    &beta,
                    dY_cublas, N);
    }
    cudaDeviceSynchronize();

    // BENCHMARK: Start timer for cuBLAS
    cudaEventRecord(start);

    // BENCHMARK: The cuBLAS SGEMM call!
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    dX, N,
                    dW, K,
                    &beta,
                    dY_cublas, N);

    // BENCHMARK: Stop timer and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_time, start, stop);

    // ------------------- Reference + Error + Timings -------------------
    std::vector<float> hY_ref(M * N);
    cudaMemcpy(hY_ref.data(), dY_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\n--- BENCHMARK RESULTS ---" << std::endl;
    printf("Custom Quantized GEMM time:  %.3f ms\n", qgemm_time);
    printf("cuBLAS FP32 GEMM time:     %.3f ms\n\n", cublas_time);
    printf("Speedup: %.3f\n", (cublas_time/qgemm_time));
    
    std::cout << "--- ACCURACY (Custom Kernel vs cuBLAS) ---" << std::endl;
    compute_mse(hY_ref, hY, M, N);
    std::cout << std::endl;

    // ------------------- Cleanup -------------------
    cudaFree(dW); cudaFree(dX); cudaFree(dWq); cudaFree(dXq);
    cudaFree(dW_scales); cudaFree(dX_colMins); cudaFree(dX_colMaxs);
    cudaFree(dX_scales); cudaFree(dX_zps); cudaFree(dY);
    cudaFree(dY_cublas);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(cublas_handle);

    return 0;
}