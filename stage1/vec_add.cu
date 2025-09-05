#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(call) \
    { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); }}

__global__ void vec_add(float* A, float* B, float* C, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < N){
        C[i] = A[i] + B[i];
    }
}

int main(){
    int N = 16;
    size_t size = N * sizeof(float);

    std::vector<float> h_A(N, 1.0f), h_B(N, 2.0f), h_C(N);

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

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    vec_add<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    std::cout << "A + B = [ ";
    for (int i = 0; i < N; ++i){
        std::cout << h_C[i] << " ";
    }
    std::cout << "]" << std::endl;

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));

    return 0;
}