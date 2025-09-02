#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 2

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

        for(int p = 0; p < k; p++){
            sum += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < m && col < n){
        C[row * n + col] = sum;
    }

}

int main(){
    int m = 4, k = 4, n = 4;
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);


    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    for (int i = 0; i < m; ++i){
        for(int j = 0; j < k; ++j){
            h_A[i*k + j] = i*0.1 + j*0.2;
            // h_A[i*k + j] = 1.0;
        }
    }
    for (int i = 0; i < k; ++i){
        for(int j = 0; j < n; ++j){
            h_B[i*n + j] = i*0.2 + j*0.1;
            // h_B[i*n + j] = 2.0;
        }
    }
    printf("A = [");
    for (int i = 0; i < m; ++i){
        printf("[ ");
        for(int j = 0; j < k; ++j){
            printf("%.2f ", h_A[i*k + j]);
        }
        printf("]\n");
    }

    printf("]\n");
    printf("B = [");
    for (int i = 0; i < m; ++i){
        printf("[ ");
        for(int j = 0; j < k; ++j){
            printf("%.2f ", h_B[i*n + j]);
        }
        printf("]\n");
    }
    printf("]\n");

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((n + TILE_SIZE - 1)/TILE_SIZE, (m + TILE_SIZE - 1)/TILE_SIZE);

    matmul_tiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("C = [");
    for (int i = 0; i < m; ++i){
        printf("[ ");
        for(int j = 0; j < k; ++j){
            printf("%.2f ", h_C[i*n + j]);
        }
        printf("]\n");
    }
    printf("]\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}