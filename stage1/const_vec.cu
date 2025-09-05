#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(call) \
    { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); }}

__global__ void const_vec(float* arr, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N){
        arr[i] = 1.0f;
    }
}

int main(){
    int N = 16;

    size_t size = N * sizeof(float);
    std::vector<float> arr_h(N);

    float* arr_d;
    CUDA_CHECK(cudaMalloc(&arr_d, size));

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1)/threadsPerBlock;

    const_vec<<<numBlocks, threadsPerBlock>>>(arr_d, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(arr_h.data(), arr_d, size, cudaMemcpyDeviceToHost));

    for(int i = 0; i < N; ++i){
        std::cout << arr_h[i] << " ";
    }
    std:: cout << std::endl;

    CUDA_CHECK(cudaFree(arr_d));

    return 0;
}