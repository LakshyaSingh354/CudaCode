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
                              int N)
{
    extern __shared__ float sdata[];
    float* smin = sdata;                         // first half for mins
    float* smax = sdata + blockDim.x;            // second half for maxs

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        smin[tid] = A[i];
        smax[tid] = A[i];
    }
    __syncthreads();

    // Reduction for min and max in parallel
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smin[tid] = fminf(smin[tid], smin[tid + stride]);
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockMins[blockIdx.x] = smin[0];
        blockMaxs[blockIdx.x] = smax[0];
    }
}


// quantization
__global__ void quantize(const float* __restrict__ A, uint8_t* __restrict__ Q, float scale, float zeroPoint, float N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < N){
        int q = __float2int_rn(A[i] / scale) + zeroPoint;
        q = max(0, min(255, q));
        Q[i] = static_cast<uint8_t>(q);
    }
}
// dequantization
__global__ void dequantize(const uint8_t* Q, float* A_rec, float scale, float zeroPoint, float N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N){
        float a = static_cast<float>(Q[i]);
        A_rec[i] = (a - zeroPoint) * scale;
    }
}

int main(){
    int N = 1 << 20;
    std::vector<float> h_A(N);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(3.0f, 0.5f);
    for (int i = 0; i < N; i++) {
        h_A[i] = dist(rng);
    }
    float* d_A;
    CHECK_CUDA(cudaMalloc(&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float* d_partial_min; CHECK_CUDA(cudaMalloc(&d_partial_min, blocks * sizeof(float)));
    float* d_partial_max; CHECK_CUDA(cudaMalloc(&d_partial_max, blocks * sizeof(float)));

    minmax_kernel<<<blocks, threads, 2 * threads * sizeof(float)>>>(d_A, d_partial_min, d_partial_max, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    std::vector<float> h_partial_min(blocks); std::vector<float> h_partial_max(blocks);
    CHECK_CUDA(cudaMemcpy(h_partial_max.data(), d_partial_max, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    float maxval = -INFINITY;
    for (float v : h_partial_max) maxval = std::max(maxval, v);
    std::cout << "Max = " << maxval << std::endl;

    CHECK_CUDA(cudaMemcpy(h_partial_min.data(), d_partial_min, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    float minval = INFINITY;
    for (float v : h_partial_min) minval = std::min(minval, v);
    std::cout << "Min = " << minval << std::endl;

    float scale = (maxval - minval) / 255.f;
    std::cout << "Scale = " << scale << std::endl;

    float zeroPoint = -(minval / scale);
    zeroPoint = max(0.0, min(zeroPoint, 255.0));
    std::cout << "Zero Point = " << zeroPoint << std::endl;

    uint8_t* d_Q;
    CHECK_CUDA(cudaMalloc(&d_Q, N * sizeof(uint8_t)));
    quantize<<<blocks, threads>>>(d_A, d_Q, scale, zeroPoint, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    float* d_Arec;
    CHECK_CUDA(cudaMalloc(&d_Arec, N * sizeof(float)));
    dequantize<<<blocks, threads>>>(d_Q, d_Arec, scale, zeroPoint, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    std::vector<float> h_Arec(N);
    CHECK_CUDA(cudaMemcpy(h_Arec.data(), d_Arec, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute error
    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = h_A[i] - h_Arec[i];
        mse += diff * diff;
    }
    mse /= N;
    std::cout << "MSE: " << mse << std::endl;

    cudaFree(d_A); cudaFree(d_partial_min); cudaFree(d_partial_max); cudaFree(d_Q); cudaFree(d_Arec);
    return 0;
}