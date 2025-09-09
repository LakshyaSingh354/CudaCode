#pragma once
#include <cuda_runtime.h>

__global__ void naive_gemm(const float* A, const float* B, float* C,
                                    int M, int K, int N, 
                                    float alpha, float beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float val = 0.0f;
        for (int k = 0; k < N; k++) {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * val + beta * C[row * N + col];
    }
}

// ------------------- Launcher wrapper -------------------
void kernel(const float* A, const float* B, float* C,
                  int M, int K, int N, 
                  float alpha, float beta) {
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naive_gemm<<<numBlocks, threadsPerBlock>>>(A, B, C, M, K, N, alpha, beta);
}
