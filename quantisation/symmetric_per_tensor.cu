#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define CHECK_CUDA(call) \
    { cudaError_t err = call; if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1);}}

// absolute max kernel
__global__ void abs_max(const float* __restrict__ A, float* maxVal, int N){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float val = 0.0f;
    if (i < N) val = fabsf(A[i]);

    sdata[tid] = val;
    __syncthreads();

    for(int j = blockDim.x / 2; j > 0; j >>= 1){
        if (tid < j) sdata[tid] = fmaxf(sdata[tid], sdata[tid + j]);
        __syncthreads();
    }

    if(tid == 0) maxVal[blockIdx.x] = sdata[0];
}

// quantization
__global__ void quantize(const float* __restrict__ A, int8_t* __restrict__ Q, float scale, float N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < N){
        int q = __float2int_rn(A[i]);
        q = max(-127, min(127, q));
        Q[i] = static_cast<int8_t>(q);
    }
}
// dequantization
__global__ void dequantize(const int8_t* Q, float* A_rec, float scale, float N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N){
        float a = static_cast<float>(Q[i]);
        A_rec[i] = a * scale;
    }
}

int main(){
    int N = 1 << 20;
    std::vector<float> h_A(N);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (int i = 0; i < N; i++) {
        h_A[i] = dist(rng);
    }
    float* d_A;
    CHECK_CUDA(cudaMalloc(&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float* d_partial; CHECK_CUDA(cudaMalloc(&d_partial, blocks * sizeof(float)));

    abs_max<<<blocks, threads, threads * sizeof(float)>>>(d_A, d_partial, N);

    std::vector<float> h_partial(blocks);
    CHECK_CUDA(cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    float maxval = 0.0f;
    for (float v : h_partial) maxval = std::max(maxval, v);
    float scale = maxval / 127.f;
    std::cout << "Scale = " << scale << std::endl;

    int8_t* d_Q;
    CHECK_CUDA(cudaMalloc(&d_Q, N * sizeof(int8_t)));
    quantize<<<blocks, threads>>>(d_A, d_Q, scale, N);
    float* d_Arec;
    CHECK_CUDA(cudaMalloc(&d_Arec, N * sizeof(float)));
    dequantize<<<blocks, threads>>>(d_Q, d_Arec, scale, N);

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

    cudaFree(d_A); cudaFree(d_partial); cudaFree(d_Q); cudaFree(d_Arec);
    return 0;
}