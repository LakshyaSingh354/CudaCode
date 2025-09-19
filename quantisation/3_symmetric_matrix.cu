#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define CHECK_CUDA(call) \
    { cudaError_t err = call; if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1);}}

__global__ void absmax_kernel(const float* __restrict__ A,
                              float* blockMaxs,
                              int M, int N)
{
    extern __shared__ float sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int m = blockIdx.y * blockDim.y + ty;
    int n = blockIdx.x * blockDim.x + tx;

    float val = INFINITY;
    if (m < M && n < N) {
        val = fabsf(A[m * N + n]);
    }
    sdata[tid] = val;
    __syncthreads();

    // reduction (1D across tid)
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int blockId = blockIdx.y * gridDim.x + blockIdx.x;
        blockMaxs[blockId] = sdata[0];
    }
}

// ---------------------------------------------------------------------
// Quantization / Dequantization
// ---------------------------------------------------------------------
__global__ void quantize(const float* __restrict__ A,
                         int8_t* __restrict__ Q,
                         float scale,
                         int M, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if(m < M && n < N){
        int q = __float2int_rn(A[m * N + n] / scale);
        q = max(-127, min(127, q));
        Q[m * N + n] = static_cast<int8_t>(q);
    }
}

__global__ void dequantize(const int8_t* Q,
                           float* A_rec,
                           float scale,
                           int M, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N){
        float a = static_cast<float>(Q[m * N + n]);
        A_rec[m * N + n] = a * scale;
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
    std::normal_distribution<float> dist(0.0f, 0.1f);
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
    float* d_partial; CHECK_CUDA(cudaMalloc(&d_partial, numBlocks * sizeof(float)));

    // Launch max kernel
    absmax_kernel<<<blocks, threads, threads.x * threads.y * sizeof(float)>>>(
        d_A, d_partial, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // Copy partial results back
    std::vector<float> h_partial(numBlocks);
    CHECK_CUDA(cudaMemcpy(h_partial.data(), d_partial,
                          numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    float maxval = -INFINITY;
    for (float v : h_partial) maxval = std::max(maxval, v);

    std::cout << "Max = " << maxval << "\n";

    // Compute scale/zero-point
    float scale = (maxval) / 127.f;
    std::cout << "Scale = " << scale << "\n";

    // Allocate quantized/dequantized buffers
    int8_t* d_Q;
    CHECK_CUDA(cudaMalloc(&d_Q, M * N * sizeof(int8_t)));
    float* d_Arec;
    CHECK_CUDA(cudaMalloc(&d_Arec, M * N * sizeof(float)));

    // Quantize + Dequantize
    quantize<<<blocks, threads>>>(d_A, d_Q, scale, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    dequantize<<<blocks, threads>>>(d_Q, d_Arec, scale, M, N);
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
    cudaFree(d_partial);
    cudaFree(d_Q);
    cudaFree(d_Arec);

    return 0;
}
