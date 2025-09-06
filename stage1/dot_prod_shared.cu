#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(call) \
    { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); }}

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

int main(){
    int N = 4096;
    size_t size = N * sizeof(float);

    std::vector<float> h_A(N, 1.5f), h_B(N, 2.5f);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t size_block = numBlocks * sizeof(float);

    std::vector<float> h_block_sum(numBlocks);

    float *d_A, *d_B, *d_block_sum;

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_block_sum, size_block));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    dot_prod<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_A, d_B, d_block_sum, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_block_sum.data(), d_block_sum, size_block, cudaMemcpyDeviceToHost));

    float dot = 0.0f;
    for (int i = 0; i < numBlocks; ++i){
        dot += h_block_sum[i];
    }

    std::cout << "A . B = " << dot << std::endl;

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_block_sum));

    return 0;
}