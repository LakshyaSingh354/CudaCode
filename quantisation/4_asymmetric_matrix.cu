#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define CHECK_CUDA(call) \
    { cudaError_t err = call; if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1);}}

__global__ void minmax_kernel(const float* __restrict__ A,
                              float* blockMins,
                              float* blockMaxs,
                              int M, int N)
{
    extern __shared__ float sdata[];
    float* smin = sdata;                         // first half for mins
    float* smax = sdata + blockDim.x * blockDim.y; // second half for maxs

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int m = blockIdx.y * blockDim.y + ty;
    int n = blockIdx.x * blockDim.x + tx;

    float val = INFINITY;
    float vax = -INFINITY;
    if (m < M && n < N) {
        val = A[m * N + n];
        vax = val;
    }
    smin[tid] = val;
    smax[tid] = vax;
    __syncthreads();

    // reduction (1D across tid)
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smin[tid] = fminf(smin[tid], smin[tid + stride]);
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int blockId = blockIdx.y * gridDim.x + blockIdx.x;
        blockMins[blockId] = smin[0];
        blockMaxs[blockId] = smax[0];
    }
}

// ---------------------------------------------------------------------
// Quantization / Dequantization
// ---------------------------------------------------------------------
__global__ void quantize(const float* __restrict__ A,
                         uint8_t* __restrict__ Q,
                         float scale, float zeroPoint,
                         int M, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if(m < M && n < N){
        int q = __float2int_rn(A[m * N + n] / scale) + zeroPoint;
        q = max(0, min(255, q));
        Q[m * N + n] = static_cast<uint8_t>(q);
    }
}

__global__ void dequantize(const uint8_t* Q,
                           float* A_rec,
                           float scale, float zeroPoint,
                           int M, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N){
        float a = static_cast<float>(Q[m * N + n]);
        A_rec[m * N + n] = (a - zeroPoint) * scale;
    }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------
int main(){
    int M = 1024;
    int N = 1024;

    // Host data
    std::vector<float> h_A(M * N);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(3.0f, 0.5f);
    for (int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++){
            h_A[i * N + j] = dist(rng);
        }
    }

    // Device memory
    float* d_A;
    CHECK_CUDA(cudaMalloc(&d_A, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                1);

    int numBlocks = blocks.x * blocks.y;
    float* d_partial_min; CHECK_CUDA(cudaMalloc(&d_partial_min, numBlocks * sizeof(float)));
    float* d_partial_max; CHECK_CUDA(cudaMalloc(&d_partial_max, numBlocks * sizeof(float)));

    // Launch min/max kernel
    minmax_kernel<<<blocks, threads, 2 * threads.x * threads.y * sizeof(float)>>>(
        d_A, d_partial_min, d_partial_max, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // Copy partial results back
    std::vector<float> h_partial_min(numBlocks), h_partial_max(numBlocks);
    CHECK_CUDA(cudaMemcpy(h_partial_min.data(), d_partial_min,
                          numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_partial_max.data(), d_partial_max,
                          numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    float minval = INFINITY;
    float maxval = -INFINITY;
    for (float v : h_partial_min) minval = std::min(minval, v);
    for (float v : h_partial_max) maxval = std::max(maxval, v);

    std::cout << "Min = " << minval << "\n";
    std::cout << "Max = " << maxval << "\n";

    // Compute scale/zero-point
    float scale = (maxval - minval) / 255.f;
    float zeroPoint = -minval / scale;
    zeroPoint = std::max(0.0f, std::min(zeroPoint, 255.0f));
    std::cout << "Scale = " << scale << "\n";
    std::cout << "Zero Point = " << zeroPoint << "\n";

    // Allocate quantized/dequantized buffers
    uint8_t* d_Q;
    CHECK_CUDA(cudaMalloc(&d_Q, M * N * sizeof(uint8_t)));
    float* d_Arec;
    CHECK_CUDA(cudaMalloc(&d_Arec, M * N * sizeof(float)));

    // Quantize + Dequantize
    quantize<<<blocks, threads>>>(d_A, d_Q, scale, zeroPoint, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    dequantize<<<blocks, threads>>>(d_Q, d_Arec, scale, zeroPoint, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // Copy back reconstructed data
    std::vector<float> h_Arec(M * N);
    CHECK_CUDA(cudaMemcpy(h_Arec.data(), d_Arec, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute error (MSE)
    double mse = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = h_A[i] - h_Arec[i];
        mse += diff * diff;
    }
    mse /= (M * N);
    std::cout << "MSE: " << mse << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_partial_min);
    cudaFree(d_partial_max);
    cudaFree(d_Q);
    cudaFree(d_Arec);

    return 0;
}
