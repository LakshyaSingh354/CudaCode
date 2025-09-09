#pragma once

#include <cuda_runtime.h>


#define BLOCK_SIZE 32

__global__ void sgemm_global_mem_coalesce(const float* A, const float* B, float* C, 
                                            int M, int K, int N, 
                                            float alpha, float beta){

    const int row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const int col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (row < M && col <  N){
        float sum = 0.0;

        for (int i = 0; i < K; ++i){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void kernel(const float* A, const float* B, float* C,
                  int M, int K, int N, 
                  float alpha, float beta) {
    dim3 block(BLOCK_SIZE * BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sgemm_global_mem_coalesce<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}