#pragma once
#include <cuda_runtime.h>

__global__ void naive_gemm(const float* A, const float* B, float* C,
                                    int N, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float val = 0.0f;
        for (int k = 0; k < N; k++) {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * val + beta * C[row * N + col];
    }
}

// ------------------- Launcher wrapper -------------------
void kernel_naive(const float* A, const float* B, float* C,
                  int N, float alpha, float beta) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naive_gemm<<<numBlocks, threadsPerBlock>>>(A, B, C, N, alpha, beta);
}
