#pragma once
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 32
#define TM 8
#define TN 8

__global__ void __launch_bounds__(256) sgemm_vec_transA (const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int K, int N,
                                 float alpha, float beta) {

    int threadRow = threadIdx.y; // 0..(BM/TM - 1)
    int threadCol = threadIdx.x; // 0..(BN/TN - 1)
    int threads_per_block = blockDim.x * blockDim.y;
    int linearThread = threadRow * blockDim.x + threadCol;

    int C_block_row = blockIdx.y * BM;
    int C_block_col = blockIdx.x * BN;
    int C_start_row = C_block_row + threadRow * TM;
    int C_start_col = C_block_col + threadCol * TN;

    float threadResults[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            threadResults[i][j] = 0.0f;

    // Notice As is transposed: BK x BM (instead of BM x BK)
    __shared__ float As[BK][BM]; // 8 x 64
    __shared__ float Bs[BK][BN]; // 8 x 64

    for (int t = 0; t < K; t += BK) {

        // --- Load As (BMxBK elements) as float4 and transpose ---
        // Each iteration loads 4 consecutive floats in row-major A
        int numAels = BM * BK;
        int loadsPerThread = (numAels + 4*threads_per_block - 1) / (4*threads_per_block);

        for (int l = 0; l < loadsPerThread; l++) {
            int loadIdx = 4 * (linearThread + l * threads_per_block);
            if (loadIdx < numAels) {
                int a_row = loadIdx / BK; // row in BM
                int a_col = loadIdx % BK; // col in BK
                int global_a_row = C_block_row + a_row;
                int global_a_col = t + a_col;

                float4 tmp;
                if (global_a_row < M && global_a_col + 3 < K) {
                    tmp = *reinterpret_cast<const float4*>(
                            &A[global_a_row * K + global_a_col]);
                } else {
                    tmp = make_float4(0.f, 0.f, 0.f, 0.f);
                }

                // store transposed into shared memory: As[col][row]
                As[a_col + 0][a_row] = tmp.x;
                if (a_col + 1 < BK) As[a_col + 1][a_row] = tmp.y;
                if (a_col + 2 < BK) As[a_col + 2][a_row] = tmp.z;
                if (a_col + 3 < BK) As[a_col + 3][a_row] = tmp.w;
            }
        }

        // --- Load Bs (BKxBN elements) as float4 ---
        int numBels = BK * BN;
        int loadsPerThreadB = (numBels + 4*threads_per_block - 1) / (4*threads_per_block);

        for (int l = 0; l < loadsPerThreadB; l++) {
            int loadIdx = 4 * (linearThread + l * threads_per_block);
            if (loadIdx < numBels) {
                int b_row = loadIdx / BN; // 0..BK-1
                int b_col = loadIdx % BN; // 0..BN-1
                int global_b_row = t + b_row;
                int global_b_col = C_block_col + b_col;

                float4 tmp;
                if (global_b_row < K && global_b_col + 3 < N) {
                    tmp = *reinterpret_cast<const float4*>(
                            &B[global_b_row * N + global_b_col]);
                } else {
                    tmp = make_float4(0.f, 0.f, 0.f, 0.f);
                }

                Bs[b_row][b_col + 0] = tmp.x;
                if (b_col + 1 < BN) Bs[b_row][b_col + 1] = tmp.y;
                if (b_col + 2 < BN) Bs[b_row][b_col + 2] = tmp.z;
                if (b_col + 3 < BN) Bs[b_row][b_col + 3] = tmp.w;
            }
        }

        __syncthreads();

        // --- Compute outer product (TMxTN) ---
        #pragma unroll
        for (int kidx = 0; kidx < BK; kidx++) {
            float Areg[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                int a_row = threadRow * TM + i;
                Areg[i] = As[kidx][a_row]; // note transpose: As[k][row]
            }

            float Breg[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int b_col = threadCol * TN + j;
                Breg[j] = Bs[kidx][b_col];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    threadResults[i][j] += Areg[i] * Breg[j];
                }
            }
        }
        __syncthreads();
    }

    // --- Write back ---
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = C_start_row + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int col = C_start_col + j;
                if (col < N) {
                    float old = C[row * N + col];
                    C[row * N + col] = alpha * threadResults[i][j] + beta * old;
                }
            }
        }
    }
}

inline void kernel(const float* A, const float* B, float* C,
                                    int M, int K, int N,
                                    float alpha, float beta) {
    dim3 block(BN / TN, BM / TM); // 8x8 = 64 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_vec_transA<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}
