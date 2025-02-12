#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matAdd(int* a, int* b, int* c, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i < N && j < N){
        c[i * N + j] = a[i * N + j] + b[i * N + j];
    }
}

__managed__ int matrix_a[1000][1000], matrix_b[1000][1000], matrix_c[1000][1000];

int main() {
    int N = 1000;
    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dim3 threads(threadsPerBlock, threadsPerBlock);
    dim3 blocks(blocksPerGrid, blocksPerGrid);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            matrix_a[i][j] = i + j;
            matrix_b[i][j] = N - i - j;
        }
    }

    matAdd<<<blocks, threads>>>(matrix_a[0], matrix_b[0], matrix_c[0], N);

    int result_sum = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            result_sum += matrix_c[i][j];
        }
    }

    return 0;
}