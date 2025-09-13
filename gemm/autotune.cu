// =================================================================================
// Autotuning script for a vectorized SGEMM CUDA Kernel
//
// How to Compile:
// nvcc -O3 -arch=sm_86 autotune.cu -o autotune
// (Change -arch=sm_86 to your GPU's compute capability, e.g., sm_75, sm_80)
//
// How to Run:
// ./autotune
// =================================================================================

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// --- Utility Macro for CUDA Error Checking ---
#define CUDA_CHECK(call)                                                 \
do {                                                                     \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA Error at %s:%d, %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)


// =================================================================================
// Templated SGEMM Kernel (Your Vectorized Implementation)
// =================================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_vec_transA(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int K, int N,
                                 float alpha, float beta) {
    // This is your Kernel 6 implementation
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    int linearThread = threadRow * blockDim.x + threadCol;

    int C_block_row = blockIdx.y * BM;
    int C_block_col = blockIdx.x * BN;
    int C_start_row = C_block_row + threadRow * TM;
    int C_start_col = C_block_col + threadCol * TN;

    float threadResults[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            threadResults[i][j] = 0.0f;

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    for (int t = 0; t < K; t += BK) {
        // Load As (transposed) and Bs using vectorized float4
        int numAels = BM * BK;
        for (int i = linearThread; i < numAels / 4; i += threads_per_block) {
            int a_row = (i * 4) / BK;
            int a_col = (i * 4) % BK;
            int global_a_row = C_block_row + a_row;
            if (global_a_row < M && (t + a_col + 3) < K) {
                float4 tmp = *reinterpret_cast<const float4*>(&A[global_a_row * K + (t + a_col)]);
                As[a_col + 0][a_row] = tmp.x;
                As[a_col + 1][a_row] = tmp.y;
                As[a_col + 2][a_row] = tmp.z;
                As[a_col + 3][a_row] = tmp.w;
            }
        }

        int numBels = BK * BN;
        for (int i = linearThread; i < numBels / 4; i += threads_per_block) {
            int b_row = (i * 4) / BN;
            int b_col = (i * 4) % BN;
            int global_b_row = t + b_row;
            int global_b_col = C_block_col + b_col;
            if (global_b_row < K && (global_b_col + 3) < N) {
                float4 tmp = *reinterpret_cast<const float4*>(&B[global_b_row * N + global_b_col]);
                Bs[b_row][b_col + 0] = tmp.x;
                Bs[b_row][b_col + 1] = tmp.y;
                Bs[b_row][b_col + 2] = tmp.z;
                Bs[b_row][b_col + 3] = tmp.w;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kidx = 0; kidx < BK; kidx++) {
            float Areg[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++) Areg[i] = As[kidx][threadRow * TM + i];

            float Breg[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) Breg[j] = Bs[kidx][threadCol * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    threadResults[i][j] += Areg[i] * Breg[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = C_start_row + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int col = C_start_col + j;
                if (col < N) C[row * N + col] = alpha * threadResults[i][j] + beta * C[row * N + col];
            }
        }
    }
}


// =================================================================================
// Kernel Wrappers for each configuration we want to test
// =================================================================================

// Pointer to a kernel function
using kernel_func_t = void (*)(const float*, const float*, float*, int, int, int, float, float);

// Config 1
void kernel_128_128_8_8_8(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta) {
    dim3 block(128 / 8, 128 / 8); // BN/TN, BM/TM
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    sgemm_vec_transA<128, 128, 8, 8, 8><<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}

// Config 2 (from author's A100 tuning)
void kernel_64_64_16_4_4(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta) {
    dim3 block(64 / 4, 64 / 4); // BN/TN, BM/TM
    dim3 grid((N + 64 - 1) / 64, (M + 64 - 1) / 64);
    sgemm_vec_transA<64, 64, 16, 4, 4><<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}

// Config 3 (larger K tile)
void kernel_128_128_16_8_8(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta) {
    dim3 block(128 / 8, 128 / 8); // BN/TN, BM/TM
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    sgemm_vec_transA<128, 128, 16, 8, 8><<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}

// Config 4 (wider tile)
void kernel_64_128_8_8_8(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta) {
    dim3 block(128 / 8, 64 / 8); // BN/TN, BM/TM
    dim3 grid((N + 128 - 1) / 128, (M + 64 - 1) / 64);
    sgemm_vec_transA<64, 128, 8, 8, 8><<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}
// Config 5 (Tall Tile)
void kernel_128_64_16_8_8(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta) {
    dim3 block(64 / 8, 128 / 8); // BN/TN, BM/TM
    dim3 grid((N + 64 - 1) / 64, (M + 128 - 1) / 128);
    sgemm_vec_transA<128, 64, 16, 8, 8><<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}

// Config 6 (Large Tile)
void kernel_256_128_8_8_8(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta) {
    dim3 block(128 / 8, 256 / 8); // BN/TN, BM/TM
    dim3 grid((N + 128 - 1) / 128, (M + 256 - 1) / 256);
    sgemm_vec_transA<256, 128, 8, 8, 8><<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}

// Config 7 (Asymmetric Thread Tile)
void kernel_64_64_8_4_8(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta) {
    dim3 block(64 / 8, 64 / 4); // BN/TN, BM/TM
    dim3 grid((N + 64 - 1) / 64, (M + 64 - 1) / 64);
    sgemm_vec_transA<64, 64, 8, 4, 8><<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}
// =================================================================================
// Benchmarking Utilities
// =================================================================================

float benchmark(kernel_func_t kernel, const float* d_A, const float* d_B, float* d_C, int N, int iters, float alpha, float beta) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 5; i++) kernel(d_A, d_B, d_C, N, N, N, alpha, beta); // Warmup
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) kernel(d_A, d_B, d_C, N, N, N, alpha, beta);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

double gflops(int N, float ms) {
    return (2.0 * N * N * N) / (ms / 1e3) / 1e9;
}

// =================================================================================
// Main Driver
// =================================================================================
int main() {
    const int N = 4096;
    const float alpha = 1.0f, beta = 0.0f;
    
    // --- Define the configurations to test ---
    struct Config {
        std::string name;
        kernel_func_t func;
    };
    
    std::vector<Config> configs = {
        {"BM=128, BN=128, BK=8, TM=8, TN=8", kernel_128_128_8_8_8},
        {"BM=64,  BN=64,  BK=16, TM=4, TN=4", kernel_64_64_16_4_4},
        {"BM=128, BN=128, BK=16, TM=8, TN=8", kernel_128_128_16_8_8},
        {"BM=64,  BN=128, BK=8,  TM=8, TN=8", kernel_64_128_8_8_8},
        {"BM=128, BN=64,  BK=16, TM=8, TN=8", kernel_128_64_16_8_8},
        {"BM=256, BN=128, BK=8,  TM=8, TN=8", kernel_256_128_8_8_8},
        {"BM=64,  BN=64,  BK=8,  TM=4, TN=8", kernel_64_64_8_4_8}
    };
    
    std::cout << "--- Autotuning for Matrix Size " << N << "x" << N << " ---" << std::endl;
    
    // --- Prepare Data (once for all benchmarks) ---
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N, 0.0f);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : h_A) x = dist(rng);
    for (auto &x : h_B) x = dist(rng);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
    
    // --- Loop, Benchmark, and Find the Best ---
    double best_perf = 0.0;
    std::string best_config = "None";
    
    for (const auto& config : configs) {
        // Reset C for each run to ensure fairness
        CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));

        float ms = benchmark(config.func, d_A, d_B, d_C, N, 20, alpha, beta);
        double perf = gflops(N, ms);
        std::cout << "Config: " << config.name << "  ->  " << perf << " GFLOPS" << std::endl;
        
        if (perf > best_perf) {
            best_perf = perf;
            best_config = config.name;
        }
    }

    // --- Report the Winner ---
    std::cout << "\n--- Best Configuration on Your GPU ---" << std::endl;
    std::cout << "Config: " << best_config << std::endl;
    std::cout << "Performance: " << best_perf << " GFLOPS" << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}