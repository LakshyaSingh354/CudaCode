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

__global__ void sum_rowwise_kernel(const int8_t* __restrict__ Wq, int32_t* __restrict__ Sumw, int M, int K) {
    extern __shared__ int32_t sdata_sum[];

    int m = blockIdx.x;
    if (m >= M) return;

    int tid = threadIdx.x;
    int32_t sum = 0;
    for (int k = tid; k < K; k += blockDim.x) {
        sum += static_cast<int32_t>(Wq[m * K + k]);
    }
    sdata_sum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_sum[tid] += sdata_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        Sumw[m] = sdata_sum[0];
    }
}

#define TILE_SIZE 64
#define REG_TILE_M 4
#define REG_TILE_N 2  // Changed to 2 to match new block size

__global__ void qgemm_kernel_register_tiled_tensor_cores(const int8_t* __restrict__ Wq,
                                                        const float* __restrict__ Sw,
                                                        const uint8_t* __restrict__ Xq,
                                                        const float* __restrict__ Sx,
                                                        const int* __restrict__ Zx,
                                                        const int32_t* __restrict__ Sumw,
                                                        float* Y, int M, int K, int N) {
    __shared__ int8_t Wq_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t Xq_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t acc_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t sumw_tile[TILE_SIZE];
    __shared__ int32_t zx_tile[TILE_SIZE];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const int base_row = by * TILE_SIZE;
    const int base_col = bx * TILE_SIZE;

    const int thread_row_in_tile = ty * REG_TILE_M;
    const int thread_col_in_tile = tx * REG_TILE_N;

    int32_t acc[REG_TILE_M][REG_TILE_N];
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i)
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j)
            acc[i][j] = 0;

    int8_t w_reg[REG_TILE_M];

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        const int K_tile_valid = min(TILE_SIZE, K - t * TILE_SIZE);

        // Cooperative load (each thread loads REG_TILE_M x REG_TILE_N entries)
        #pragma unroll
        for (int lm = 0; lm < REG_TILE_M; ++lm) {
            #pragma unroll
            for (int ln = 0; ln < REG_TILE_N; ++ln) {
                int tile_r = thread_row_in_tile + lm;
                int tile_c = thread_col_in_tile + ln;

                int Wq_row = base_row + tile_r;
                int Wq_col = t * TILE_SIZE + tile_c;
                if (Wq_row < M && Wq_col < K) {
                    Wq_tile[tile_r][tile_c] = Wq[Wq_row * K + Wq_col];
                } else {
                    Wq_tile[tile_r][tile_c] = 0;
                }

                int Xq_row = t * TILE_SIZE + tile_r;
                int Xq_col = base_col + tile_c;
                if (Xq_row < K && Xq_col < N) {
                    int32_t xval = static_cast<int32_t>(Xq[Xq_row * N + Xq_col]);
                    Xq_tile[tile_r][tile_c] = xval - Zx[Xq_col];
                } else {
                    Xq_tile[tile_r][tile_c] = 0;
                }
            }
        }

        __syncthreads();

        // Compute over k dimension
        for (int k = 0; k < TILE_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < REG_TILE_M; ++i) {
                w_reg[i] = Wq_tile[thread_row_in_tile + i][k];
            }

            #pragma unroll
            for (int i = 0; i < REG_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_TILE_N; ++j) {
                    acc[i][j] += static_cast<int32_t>(w_reg[i]) * Xq_tile[k][thread_col_in_tile + j];
                }
            }
        }

        __syncthreads();
    }

    // Store accumulated results to shared memory
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) {
            acc_tile[thread_row_in_tile + i][thread_col_in_tile + j] = acc[i][j];
        }
    }

    __syncthreads();

    // Load Sumw and Zx for the tile
    for (int r = threadIdx.y * blockDim.x + threadIdx.x; r < TILE_SIZE; r += blockDim.x * blockDim.y) {
        if (r < TILE_SIZE) {
            int global_r = base_row + r;
            sumw_tile[r] = (global_r < M) ? Sumw[global_r] : 0;
            int global_c = base_col + r;
            zx_tile[r] = (global_c < N) ? Zx[global_c] : 0;
        }
    }

    __syncthreads();

    // Write back with correction and scaling
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) {
            int out_r = base_row + thread_row_in_tile + i;
            int out_c = base_col + thread_col_in_tile + j;
            if (out_r < M && out_c < N) {
                int32_t raw_acc = acc_tile[thread_row_in_tile + i][thread_col_in_tile + j];
                int32_t correction = sumw_tile[thread_row_in_tile + i] * zx_tile[thread_col_in_tile + j];
                Y[out_r * N + out_c] = static_cast<float>(raw_acc - correction) * Sw[out_r] * Sx[out_c];
            }
        }
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
    int32_t* dSumw;
    cudaMalloc(&dW_scales, M * sizeof(float));
    cudaMalloc(&dX_colMins, N * sizeof(float));
    cudaMalloc(&dX_colMaxs, N * sizeof(float));
    cudaMalloc(&dX_scales, N * sizeof(float));
    cudaMalloc(&dX_zps, N * sizeof(int));
    cudaMalloc(&dSumw, M * sizeof(int32_t));

    float* dY;
    cudaMalloc(&dY, M * N * sizeof(float));

    // After you pick TILE_SIZE, REG_TILE_M, REG_TILE_N ensure divisibility:
    assert(TILE_SIZE % REG_TILE_M == 0 && TILE_SIZE % REG_TILE_N == 0);

    // block dims:
    dim3 block(TILE_SIZE / REG_TILE_N, TILE_SIZE / REG_TILE_M);  // Now resolves to (32, 16)
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);  // Unchanged



    // Warmup (not timed)
        absmax_rowwise_kernel<<<M, 256, 256 * sizeof(float)>>>(dW, dW_scales, M, K);
        quantize_weights_rowwise<<<dim3(M, (K + 255) / 256), 256>>>(dW, dWq, dW_scales, M, K);
        colwise_minmax<<<N, 256, 2 * 256 * sizeof(float)>>>(dX, dX_colMins, dX_colMaxs, K, N);
        quantize_activations_colwise<<<dim3(N, (K + 255) / 256), 256>>>(dX, dXq, dX_colMins, dX_colMaxs, dX_scales, dX_zps, K, N);
        sum_rowwise_kernel<<<M, 256, 256 * sizeof(int32_t)>>>(dWq, dSumw, M, K);
    for (int i = 0; i < warmup_iters; ++i) {
        qgemm_kernel_register_tiled_tensor_cores<<<grid, block>>>(dWq, dW_scales, dXq, dX_scales, dX_zps, dSumw, dY, M, K, N);
    }
    cudaDeviceSynchronize();

    // Timed run
    absmax_rowwise_kernel<<<M, 256, 256 * sizeof(float)>>>(dW, dW_scales, M, K);
    quantize_weights_rowwise<<<dim3(M, (K + 255) / 256), 256>>>(dW, dWq, dW_scales, M, K);
    colwise_minmax<<<N, 256, 2 * 256 * sizeof(float)>>>(dX, dX_colMins, dX_colMaxs, K, N);
    quantize_activations_colwise<<<dim3(N, (K + 255) / 256), 256>>>(dX, dXq, dX_colMins, dX_colMaxs, dX_scales, dX_zps, K, N);
    sum_rowwise_kernel<<<M, 256, 256 * sizeof(int32_t)>>>(dWq, dSumw, M, K);
    cudaEventRecord(start);
    qgemm_kernel_register_tiled_tensor_cores<<<grid, block>>>(dWq, dW_scales, dXq, dX_scales, dX_zps, dSumw, dY, M, K, N);
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
                    dY_cublas, N);  // C^T: N x M;
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
    printf("cuBLAS FP32 GEMM time:     %.3f ms\n", cublas_time);
    printf("Speedup: %.3f\n\n", (cublas_time/qgemm_time));
    
    std::cout << "--- ACCURACY (Custom Kernel vs cuBLAS) ---" << std::endl;
    compute_mse(hY_ref, hY, M, N);
    std::cout << std::endl;

    // ------------------- Cleanup -------------------
    cudaFree(dW); cudaFree(dX); cudaFree(dWq); cudaFree(dXq);
    cudaFree(dW_scales); cudaFree(dX_colMins); cudaFree(dX_colMaxs);
    cudaFree(dX_scales); cudaFree(dX_zps); cudaFree(dSumw); cudaFree(dY);
    cudaFree(dY_cublas);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(cublas_handle);

    return 0;
}