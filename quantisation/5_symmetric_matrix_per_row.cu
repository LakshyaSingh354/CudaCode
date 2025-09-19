#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define CHECK_CUDA(call) \
    { cudaError_t err = call; if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1);}}

__global__ void absmax_rowwise_kernel(const float* __restrict__ A,
                                      float* rowMaxs,
                                      int M, int N)
{
    extern __shared__ float sdata[];

    int m = blockIdx.x;  // one block per row
    int tx = threadIdx.x;

    if (m >= M) return;

    float localMax = -INFINITY;
    for (int i = tx; i < N; i += blockDim.x) {
        localMax = fmaxf(localMax, fabsf(A[m * N + i]));
    }

    sdata[tx] = localMax;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            sdata[tx] = fmaxf(sdata[tx], sdata[tx + stride]);
        }
        __syncthreads();
    }

    if (tx == 0) {
        rowMaxs[m] = sdata[0];
    }
}


// ---------------------------------------------------------------------
// Quantization / Dequantization
// ---------------------------------------------------------------------
__global__ void quantize(const float* __restrict__ A,
                         int8_t* __restrict__ Q,
                         float* scale,
                         int M, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if(m < M && n < N){
        int q = __float2int_rn(A[m * N + n] / scale[m]);
        q = max(-127, min(127, q));
        Q[m * N + n] = static_cast<int8_t>(q);
    }
}

__global__ void dequantize(const int8_t* Q,
                           float* A_rec,
                           float* scale,
                           int M, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N){
        float a = static_cast<float>(Q[m * N + n]);
        A_rec[m * N + n] = a * scale[m];
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
    float* d_rowMax; CHECK_CUDA(cudaMalloc(&d_rowMax, M * sizeof(float)));

    // Launch max kernel
    int threadsPerBlock = 256;
    absmax_rowwise_kernel<<<M, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_A, d_rowMax, M, N);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // Copy partial results back
    std::vector<float> h_rowMax(numBlocks), h_scales(M);
    CHECK_CUDA(cudaMemcpy(h_rowMax.data(), d_rowMax,
                          M * sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 0; i < M; ++i){
        h_scales[i] = h_rowMax[i] / 127.f;
    }
    float *d_scales; CHECK_CUDA(cudaMalloc(&d_scales, M * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_scales, h_scales.data(), M*sizeof(float), cudaMemcpyHostToDevice));

    // Allocate quantized/dequantized buffers
    int8_t* d_Q;
    CHECK_CUDA(cudaMalloc(&d_Q, M * N * sizeof(int8_t)));
    float* d_Arec;
    CHECK_CUDA(cudaMalloc(&d_Arec, M * N * sizeof(float)));

    // Quantize + Dequantize
    quantize<<<blocks, threads>>>(d_A, d_Q, d_scales, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    dequantize<<<blocks, threads>>>(d_Q, d_Arec, d_scales, M, N);
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
    cudaFree(d_rowMax);
    cudaFree(d_Q);
    cudaFree(d_Arec);
    cudaFree(d_scales);

    return 0;
}
