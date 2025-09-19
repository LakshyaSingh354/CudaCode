#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define CHECK_CUDA(call) \
    { cudaError_t err = call; if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1);}}

__global__ void colwise_minmax(const float* __restrict__ A,
                              float* colMins,
                              float* colMaxs,
                              int M, int N)
{
    extern __shared__ float sdata[];
    float* smin = sdata;                         // first half for mins
    float* smax = sdata + blockDim.x; // second half for maxs

    int col = blockIdx.x;
    int tid = threadIdx.x;


    float localMin = INFINITY;
    float localMax = -INFINITY;
    for (int i = tid; i < M; i += blockDim.x) {
        float val = A[i * N + col];
        localMin = fminf(localMin, val);
        localMax = fmaxf(localMax, val);
    }
    smin[tid] = localMin;
    smax[tid] = localMax;
    __syncthreads();

    // reduction (1D across tid)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smin[tid] = fminf(smin[tid], smin[tid + stride]);
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        colMins[col] = smin[0];
        colMaxs[col] = smax[0];
    }
}

// ---------------------------------------------------------------------
// Quantization / Dequantization
// ---------------------------------------------------------------------
__global__ void quantize(const float* __restrict__ A,
                         uint8_t* __restrict__ Q,
                         float* scale, int* zeroPoint,
                         int M, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if(m < M && n < N){
        int q = __float2int_rn(A[m * N + n] / scale[n]) + zeroPoint[n];
        q = max(0, min(255, q));
        Q[m * N + n] = static_cast<uint8_t>(q);
    }
}

__global__ void dequantize(const uint8_t* Q,
                           float* A_rec,
                           float* scale, int* zeroPoint,
                           int M, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N){
        float a = static_cast<float>(Q[m * N + n]);
        A_rec[m * N + n] = (a - zeroPoint[n]) * scale[n];
    }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------
int main(){
    int M = 1024;
    int N = 1024;

    // Host data
    std::vector<float> h_A(M * N);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(3.0f, 0.5f);
    for (int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++){
            h_A[i * N + j] = dist(rng);
        }
    }

    // Device memory
    float* d_A;
    CHECK_CUDA(cudaMalloc(&d_A, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    float* d_colMin; CHECK_CUDA(cudaMalloc(&d_colMin, N * sizeof(float)));
    float* d_colMax; CHECK_CUDA(cudaMalloc(&d_colMax, N * sizeof(float)));

    // Launch min/max kernel
    int threadsPerBlock = 256;
    colwise_minmax<<<N, threadsPerBlock, 2*threadsPerBlock*sizeof(float)>>>(d_A, d_colMin, d_colMax, M, N);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // Copy partial results back
    std::vector<float> h_colMin(N), h_colMax(N);
    CHECK_CUDA(cudaMemcpy(h_colMin.data(), d_colMin,
                          N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_colMax.data(), d_colMax,
                          N * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> h_scale(N);
    std::vector<int>   h_zeroPointInt(N);

    for (int j = 0; j < N; ++j) {
        float colMin = h_colMin[j];
        float colMax = h_colMax[j];
        float s = (colMax - colMin) / 255.0f;
        if (s == 0.0f || !std::isfinite(s)) {
            // column is constant -> represent everything as that constant
            s = 1.0f;
        } else {
            float zp = -colMin / s;
            int zpi = std::round(zp);
            h_zeroPointInt[j] = zpi;
        }
        h_scale[j] = s;
    }

    float *d_scale;
    int   *d_zeroPoint;
    CHECK_CUDA(cudaMalloc(&d_scale, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_zeroPoint, N*sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_scale, h_scale.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_zeroPoint, h_zeroPointInt.data(), N*sizeof(int), cudaMemcpyHostToDevice));


    // Allocate quantized/dequantized buffers
    uint8_t* d_Q;
    CHECK_CUDA(cudaMalloc(&d_Q, M * N * sizeof(uint8_t)));
    float* d_Arec;
    CHECK_CUDA(cudaMalloc(&d_Arec, M * N * sizeof(float)));

    // Kernel launch configuration
    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                1);

    // Quantize + Dequantize
    quantize<<<blocks, threads>>>(d_A, d_Q, d_scale, d_zeroPoint, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    dequantize<<<blocks, threads>>>(d_Q, d_Arec, d_scale, d_zeroPoint, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // Copy back reconstructed data
    std::vector<float> h_Arec(M * N);
    CHECK_CUDA(cudaMemcpy(h_Arec.data(), d_Arec, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute error (MSE)
    double mse = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = h_A[i] - h_Arec[i];
        mse += diff * diff;
    }
    mse /= (M * N);
    std::cout << "MSE: " << mse << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_colMin);
    cudaFree(d_colMax);
    cudaFree(d_Q);
    cudaFree(d_Arec);
    cudaFree(d_scale);
    cudaFree(d_zeroPoint);

    return 0;
}
