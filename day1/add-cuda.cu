#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int* a, int* b, int* c) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];
}

__managed__ int vector_a[1000000000], vector_b[1000000000], vector_c[1000000000];

int main() {
    int N = 1000 * 1000000;  // 10 million elements
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Initialize vectors
    for(int i = 0; i < N; i++){
        vector_a[i] = i;
        vector_b[i] = N - i;
    }

    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // Start timing
    add<<<blocksPerGrid, threadsPerBlock>>>(vector_a, vector_b, vector_c);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);  // Stop timing

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Compute sum
    int result_sum = 0;
    for(int i = 0; i < N; i++){
        result_sum += vector_c[i];
    }
    printf("CUDA Result: sum = %d, Time = %f ms\n", result_sum, milliseconds);

    return 0;
}