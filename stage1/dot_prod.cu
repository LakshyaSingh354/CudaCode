#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(call) \
    { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); }}

__global__ void dot_prod(float* A, float* B, float* prod, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        atomicAdd(prod, A[i] * B[i]);
    }
}

int main(){
    int N = 16;
    size_t size = N * sizeof(float);

    std::vector<float> h_A(N, 1.5f), h_B(N, 2.5f);
    float h_prod = 0.0f;

    std::cout << "A = [ ";
    for (int i = 0; i < N; ++i){
        std::cout << h_A[i] << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "B = [ ";
    for (int i = 0; i < N; ++i){
        std::cout << h_B[i] << " ";
    }
    std::cout << "]" << std::endl;

    float *d_A, *d_B, *d_prod;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_prod, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_prod, 0, sizeof(float)));

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    dot_prod<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_prod, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_prod, d_prod, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "A . B = " << h_prod << std::endl;

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_prod));

    return 0;
}