#pragma once
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 32
#define TM 8
#define TN 8
#define WARP_M 32
#define WARP_N 32

__global__ void __launch_bounds__(256) sgemm_vec_transA_warptiled(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int K, int N,
                                           float alpha, float beta) {

    // --- WARPTILE: Step 1: Define Warp and Lane IDs ---
    // Total threads per block: (BN/TN) * (BM/TM) -> 16 * 16 = 256
    // Total warps per block: 256 / 32 = 8
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int laneId = (threadIdx.y * blockDim.x + threadIdx.x) % 32;

    // We have 8 warps. Let's arrange them in a 2x4 grid to tile the block.
    // Each warp will handle a WARP_M x WARP_N sub-tile of the block's work.
    int warps_per_row = (BN / WARP_N); // 128 / 64 = 2
    int warpRow = warpId / warps_per_row; // Warp's row ID within the block (0..3)
    int warpCol = warpId % warps_per_row; // Warp's col ID within the block (0..1)

    // A warp has 32 threads (lanes). Let's arrange them in a 4x8 grid.
    int laneRow = laneId / 8; // Lane's row ID within the warp (0..3)
    int laneCol = laneId % 8; // Lane's col ID within the warp (0..7)
    
    // --- Original thread-level calculations remain the same ---
    int C_block_row = blockIdx.y * BM;
    int C_block_col = blockIdx.x * BN;
    
    float threadResults[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            threadResults[i][j] = 0.0f;

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    for (int t = 0; t < K; t += BK) {
        // --- This entire GMEM loading section remains unchanged ---
        int threads_per_block = blockDim.x * blockDim.y;
        int linearThread = threadIdx.y * blockDim.x + threadIdx.x;
        // ... (your existing, correct code for loading As and Bs) ...
        int numAels = BM*BK; for(int i=linearThread;i<numAels/4;i+=threads_per_block){int a_row=(i*4)/BK,a_col=(i*4)%BK;int g_a_row=C_block_row+a_row;if(g_a_row<M&&(t+a_col+3)<K){float4 tmp=*(const float4*)(&A[g_a_row*K+(t+a_col)]);As[a_col][a_row]=tmp.x;As[a_col+1][a_row]=tmp.y;As[a_col+2][a_row]=tmp.z;As[a_col+3][a_row]=tmp.w;}}
        int numBels = BK*BN; for(int i=linearThread;i<numBels/4;i+=threads_per_block){int b_row=(i*4)/BN,b_col=(i*4)%BN;int g_b_row=t+b_row,g_b_col=C_block_col+b_col;if(g_b_row<K&&(g_b_col+3)<N){float4 tmp=*(const float4*)(&B[g_b_row*N+g_b_col]);Bs[b_row][b_col]=tmp.x;Bs[b_row][b_col+1]=tmp.y;Bs[b_row][b_col+2]=tmp.z;Bs[b_row][b_col+3]=tmp.w;}}

        __syncthreads();

        // --- WARPTILE: Step 2: Rework the computation loop ---
        // The old `kidx` loop is now the outer loop of the computation phase.
        // We add new loops inside to iterate over the warp tile.
        #pragma unroll
        for (int kidx = 0; kidx < BK; kidx++) {
            
            // Each thread is responsible for a TMxTN tile.
            // A 4x8 group of lanes (a warp) is responsible for a (4*TM)x(8*TN) tile.
            // This is WARP_M x WARP_N = (32x64) - slightly different from our target of 64x64.
            // For simplicity, we'll keep the logic as loading TM and TN values at once.
            // The key change is the indexing into shared memory.

            float Areg[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                // Calculate row in As based on warp and lane position
                int a_row = warpRow * WARP_M + laneRow * TM + i;
                Areg[i] = As[kidx][a_row];
            }

            float Breg[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                // Calculate col in Bs based on warp and lane position
                int b_col = warpCol * WARP_N + laneCol * TN + j;
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

    // --- WARPTILE: Step 3: Rework the write-back indexing ---
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        // Calculate row in C based on warp and lane position
        int row = C_block_row + warpRow * WARP_M + laneRow * TM + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                // Calculate col in C based on warp and lane position
                int col = C_block_col + warpCol * WARP_N + laneCol * TN + j;
                if (col < N) {
                    C[row * N + col] = alpha * threadResults[i][j] + beta * C[row * N + col];
                }
            }
        }
    }
}

inline void kernel(const float* A, const float* B, float* C,
                                    int M, int K, int N,
                                    float alpha, float beta) {
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_vec_transA_warptiled<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}
