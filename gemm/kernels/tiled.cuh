#pragma once

#include <cuda_runtime.h>


#define TILE_SIZE 16

__global__ void sgemm_tiled(const float* A, const float* B, float* C, 
                                            int M, int K, int N, 
                                            float alpha, float beta){
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for(int t = 0; t < (K + TILE_SIZE - 1)/TILE_SIZE; ++t){
        int Arow = row;
        int Acol = t * TILE_SIZE + threadIdx.x;
        int Brow = t * TILE_SIZE + threadIdx.y;
        int Bcol = col;

        if (Arow < M && Acol < K){
            As[threadIdx.y][threadIdx.x] = A[Arow * K + Acol];
        } else{
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (Brow < K && Bcol < N){
            Bs[threadIdx.y][threadIdx.x] = B[Brow * N + Bcol];
        } else{
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for(int p = 0; p < TILE_SIZE; p++){
            sum += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < M && col < N){
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void kernel(const float* A, const float* B, float* C,
                  int M, int K, int N, 
                  float alpha, float beta) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    sgemm_tiled<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}