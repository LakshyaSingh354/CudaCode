#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <numeric> // for std::accumulate

#define CUDA_CHECK(call) \
    { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); }}

// =======================
// Naive atomic kernel
// =======================
__global__ void dot_prod_naive(float* A, float* B, float* prod, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        atomicAdd(prod, A[i] * B[i]);
    }
}

// =======================
// Optimized reduction kernel
// =======================
__global__ void dot_prod(float* A, float* B, float* block_sum, int N){
    extern __shared__ float sdata[];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = 0.0f;
    if (i < N) val = A[i] * B[i];
    sdata[tid] = val;

    __syncthreads();

    // reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        block_sum[blockIdx.x] = sdata[0];
    }
}

// =======================
// Timing helper
// =======================
float benchmark_naive(float* d_A, float* d_B, int N, int threadsPerBlock, int numRuns){
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_prod;
    CUDA_CHECK(cudaMalloc(&d_prod, sizeof(float)));

    // Warmup
    CUDA_CHECK(cudaMemset(d_prod, 0, sizeof(float)));
    dot_prod_naive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_prod, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float totalTime = 0.0f;
    for (int i = 0; i < numRuns; i++) {
        CUDA_CHECK(cudaMemset(d_prod, 0, sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));
        dot_prod_naive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_prod, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalTime += ms;
    }

    float avgTime = totalTime / numRuns;

    // Validate
    float h_prod;
    CUDA_CHECK(cudaMemcpy(&h_prod, d_prod, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[Naive] Dot product = " << h_prod << ", Avg time = " << avgTime << " ms\n";

    CUDA_CHECK(cudaFree(d_prod));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return avgTime;
}

float benchmark_reduction(float* d_A, float* d_B, int N, int threadsPerBlock, int numRuns){
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_block_sum;
    CUDA_CHECK(cudaMalloc(&d_block_sum, numBlocks * sizeof(float)));
    std::vector<float> h_block_sum(numBlocks);

    // Warmup
    dot_prod<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_A, d_B, d_block_sum, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float totalTime = 0.0f;
    for (int i = 0; i < numRuns; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        dot_prod<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_A, d_B, d_block_sum, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalTime += ms;
    }

    float avgTime = totalTime / numRuns;

    // Validate
    CUDA_CHECK(cudaMemcpy(h_block_sum.data(), d_block_sum, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    float h_sum = std::accumulate(h_block_sum.begin(), h_block_sum.end(), 0.0f);
    std::cout << "[Reduction] Dot product = " << h_sum << ", Avg time = " << avgTime << " ms\n";

    CUDA_CHECK(cudaFree(d_block_sum));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return avgTime;
}

// =======================
// Main
// =======================
int main(){
    int N = 1 << 20; // 1M elements
    int threadsPerBlock = 256;
    int numRuns = 0;

    size_t size = N * sizeof(float);
    std::vector<float> h_A(N, 1.5f), h_B(N, 2.5f);

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    std::cout << "Benchmarking with N = " << N << ", threadsPerBlock = " << threadsPerBlock << "\n";
    float t1 = benchmark_naive(d_A, d_B, N, threadsPerBlock, numRuns);
    float t2 = benchmark_reduction(d_A, d_B, N, threadsPerBlock, numRuns);

    std::cout << "Speedup (naive / reduction): " << (t1 / t2) << "x\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return 0;
}