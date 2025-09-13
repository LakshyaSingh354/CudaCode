#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define BM 128   // Block tile size M
#define BN 128   // Block tile size N
#define BK 32    // Block tile size K (depth per iter)
#define TM 8    // Thread tile size M (rows per thread)
#define TN 8    // Thread tile size N (cols per thread)

__global__ void sgemm_2D_tiling(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int K, int N,
                                float alpha, float beta) {

    // blockDim.x = BN / TN, blockDim.y = BM / TM
    int threadRow = threadIdx.y; // 0 .. (BM/TM - 1)
    int threadCol = threadIdx.x; // 0 .. (BN/TN - 1)
    int threads_per_block = blockDim.x * blockDim.y;
    int linearThread = threadRow * blockDim.x + threadCol; // 0..(#threads-1)

    // Starting coordinates (top-left) of this thread's TMxTN subtile inside the BMxBN block
    int C_block_row = blockIdx.y * BM;
    int C_block_col = blockIdx.x * BN;
    int C_start_row = C_block_row + threadRow * TM;
    int C_start_col = C_block_col + threadCol * TN;

    // Per-thread register accumulator TM x TN
    float threadResults[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            threadResults[i][j] = 0.0f;

    // Shared memory tiles
    __shared__ float As[BM][BK]; // 64 x 8
    __shared__ float Bs[BK][BN]; // 8 x 64

    // Loop over K in chunks of BK
    for (int t = 0; t < K; t += BK) {

        // --- Cooperative load into As (BM x BK) ---
        // There are BM*BK elements; each thread loads multiple elements in a strided loop.
        int numAels = BM * BK; // 64*8 = 512
        for (int loadIdx = linearThread; loadIdx < numAels; loadIdx += threads_per_block) {
            int a_row = loadIdx / BK; // 0..BM-1
            int a_col = loadIdx % BK; // 0..BK-1
            int global_a_row = C_block_row + a_row; // which row of A
            int global_a_col = t + a_col;           // which col (K-dim) of A
            if (global_a_row < M && global_a_col < K) {
                As[a_row][a_col] = A[global_a_row * K + global_a_col];
            } else {
                As[a_row][a_col] = 0.0f;
            }
        }

        // --- Cooperative load into Bs (BK x BN) ---
        int numBels = BK * BN; // 8*64 = 512
        for (int loadIdx = linearThread; loadIdx < numBels; loadIdx += threads_per_block) {
            int b_row = loadIdx / BN; // 0..BK-1
            int b_col = loadIdx % BN; // 0..BN-1
            int global_b_row = t + b_row; // K-dim row for B
            int global_b_col = C_block_col + b_col; // N-dim col for B
            if (global_b_row < K && global_b_col < N) {
                Bs[b_row][b_col] = B[global_b_row * N + global_b_col];
            } else {
                Bs[b_row][b_col] = 0.0f;
            }
        }

        __syncthreads();

        // --- Compute: outer-product updates for this BK chunk ---
        // For each k in 0..BK-1, form Areg (TM×1) and Breg (1×TN) and do TM×TN update.
        #pragma unroll
        for (int kidx = 0; kidx < BK; ++kidx) {
            // Load Areg: TM elements from As corresponding to thread's rows
            float Areg[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int a_row = threadRow * TM + i; // 0..BM-1
                Areg[i] = As[a_row][kidx];
            }

            // Load Breg: TN elements from Bs corresponding to thread's cols
            float Breg[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int b_col = threadCol * TN + j; // 0..BN-1
                Breg[j] = Bs[kidx][b_col];
            }

            // Outer product update: TM x TN
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults[i][j] += Areg[i] * Breg[j];
                }
            }
        }

        __syncthreads(); // ensure no SMEM reuse until all threads done with this chunk
    }

    // --- Write back TM x TN results to global memory C ---
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = C_start_row + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int col = C_start_col + j;
                if (col < N) {
                    float old = C[row * N + col];
                    C[row * N + col] = alpha * threadResults[i][j] + beta * old;
                }
            }
        }
    }
}

// Host wrapper that launches the kernel
inline void kernel(const float* A, const float* B, float* C,
                            int M, int K, int N,
                            float alpha, float beta) {

    // block dims = BN/TN x BM/TM
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Launch
    sgemm_2D_tiling<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}