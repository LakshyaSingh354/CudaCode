#pragma once

#include <cuda_runtime.h>

// Step 1: Define new tiling parameters for more control
#define BM 64  // Block tile size for M dimension
#define BN 64  // Block tile size for N dimension
#define BK 8   // Block tile size for K dimension
#define TM 8   // Thread tile size for M dimension

__global__ void sgemm_1D_tiling(const float* A, const float* B, float* C, 
                                int M, int K, int N, 
                                float alpha, float beta){
    // Step 2: Adjust thread-to-matrix mapping
    // Each thread block computes a BM x BN tile of C.
    // Each thread computes a TM x 1 strip of that tile.
    int threadRow = threadIdx.y; // Row within the thread block (0 to 7)
    int threadCol = threadIdx.x; // Col within the thread block (0 to 63)

    // linear thread id inside block (0..511)
    int tIdx = threadRow * blockDim.x + threadCol;
    
    // Starting row in C for this thread's computations
    int C_start_row = blockIdx.y * BM + threadRow * TM;
    // Column in C for this thread's computations
    int C_col = blockIdx.x * BN + threadCol;

    // Step 3: Allocate a register array for the thread's results
    float threadResults[TM];
    #pragma unroll
    for (int i = 0; i < TM; ++i) threadResults[i] = 0.0f;

    // Define shared memory tiles with new dimensions
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Main loop over tiles in the K dimension
    for(int t = 0; t < K; t += BK){
        // --- Coalesced Loading from Global to Shared Memory ---
        // Each thread now loads multiple elements to fill the larger tiles.
        // This part is tricky, we'll make each thread load one element of As and one of Bs.
        // There are (BM/TM)*BN = 8*64 = 512 threads.
        // As is 64x8=512 floats. Bs is 8x64=512 floats. So 1-to-1 load is perfect.
        // Map tIdx to As: As has BM*BK = 64*8 = 512 elements
        int a_row = tIdx / BK;        // 0..63
        int a_col = tIdx % BK;        // 0..7
        int global_a_row = blockIdx.y * BM + a_row; // actual row in A
        int global_a_col = t + a_col;               // actual col in A (K-dim)

        if (global_a_row < M && global_a_col < K) {
            As[a_row][a_col] = A[global_a_row * K + global_a_col];
        } else {
            As[a_row][a_col] = 0.0f;
        }

        // Map tIdx to Bs: Bs has BK*BN = 8*64 = 512 elements
        int b_row = tIdx / BN;        // 0..7
        int b_col = tIdx % BN;        // 0..63
        int global_b_row = t + b_row;          // K-dim row in B
        int global_b_col = blockIdx.x * BN + b_col; // N-dim col in B

        if (global_b_row < K && global_b_col < N) {
            Bs[b_row][b_col] = B[global_b_row * N + global_b_col];
        } else {
            Bs[b_row][b_col] = 0.0f;
        }

        __syncthreads();

        // Step 4: Rework the inner loop for massive register reuse
        #pragma unroll
        for(int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Cache a value from Bs that will be used TM times
            float Btmp = Bs[dotIdx][threadCol];
            
            // Loop over the TM results this thread is responsible for
            #pragma unroll
            for(int resIdx = 0; resIdx < TM; ++resIdx) {
                // Accumulate into the register array
                float Aelem = As[threadRow * TM + resIdx][dotIdx];
                threadResults[resIdx] += Aelem * Btmp;
            }
        }
        __syncthreads();
    }

    // Step 5: Write the entire block of results back to global memory
    for(int i = 0; i < TM; ++i){
        int final_row = C_start_row + i;
        if (final_row < M && C_col < N) {
            float old = C[final_row * N + C_col];
            C[final_row * N + C_col] = alpha * threadResults[i] + beta * old;
        }
    }
}

void kernel(const float* A, const float* B, float* C,
                  int M, int K, int N, 
                  float alpha, float beta) {
    // Update block and grid dimensions for the new tiling strategy
    dim3 block(BN, BM / TM); // 64x8 threads per block
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    sgemm_1D_tiling<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}