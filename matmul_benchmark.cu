#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define TILE_SIZE 16

__global__ void matmul_naive(float* A, float* B, float* C, int m, int k, int n){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < m && col < n){
        float sum = 0.0f;
        for (int p = 0; p < k; ++p){
            sum += A[row * k + p] * B[p * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matmul_tiled(float* A, float* B, float* C, int m, int k, int n){
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for(int t = 0; t < (k + TILE_SIZE - 1)/TILE_SIZE; ++t){
        int Arow = row;
        int Acol = t * TILE_SIZE + threadIdx.x;
        int Brow = t * TILE_SIZE + threadIdx.y;
        int Bcol = col;

        if (Arow < m && Acol < k){
            As[threadIdx.y][threadIdx.x] = A[Arow * k + Acol];
        } else{
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (Brow < k && Bcol < n){
            Bs[threadIdx.y][threadIdx.x] = B[Brow * n + Bcol];
        } else{
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for(int p = 0; p < TILE_SIZE; p++){
            sum += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < m && col < n){
        C[row * n + col] = sum;
    }

}
void benchmark(int m, int k, int n) {
    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    // host data
    std::vector<float> h_A(m * k, 1.0f);
    std::vector<float> h_B(k * n, 2.0f);
    std::vector<float> h_C(m * n);

    // device data
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    // block & grid config
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE,
                 (m + TILE_SIZE - 1) / TILE_SIZE);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup (to avoid cold start overheads)
    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);
    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();

    // --- Benchmark naive ---
    cudaEventRecord(start);
    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);

    // --- Benchmark tiled ---
    cudaEventRecord(start);
    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled = 0;
    cudaEventElapsedTime(&ms_tiled, start, stop);

    std::cout << "Naive time: " << ms_naive << " ms\n";
    std::cout << "Tiled time: " << ms_tiled << " ms\n";
    std::cout << "Speedup: " << ms_naive/ms_tiled << " ms\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int m = 4096, k = 4096, n = 4096;
    benchmark(m, k, n);
    return 0;
}