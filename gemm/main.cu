#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "kernels/blocktiling_2d.cuh"
#include "kernels/cublas.cuh"

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

// ------------------- CPU reference GEMM -------------------
void gemm_cpu(const float* A, const float* B, float* C,
              int N, float alpha, float beta) {
    std::vector<float> result(N * N, 0.0f);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            result[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
    std::copy(result.begin(), result.end(), C);
}

// ------------------- Timer utility -------------------
float benchmark(void (*kernel)(const float*, const float*, float*, int, int, int, float, float),
                const float* d_A, const float* d_B, float* d_C,
                int N, int iters, float alpha, float beta) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 5; i++) {
        kernel(d_A, d_B, d_C, N, N, N, alpha, beta);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        kernel(d_A, d_B, d_C, N, N, N, alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;  // average time in ms
}

// ------------------- GFLOPS calculator -------------------
double gflops(int N, float ms) {
    double ops = 2.0 * N * N * N;
    double sec = ms / 1e3;
    return (ops / sec / 1e9);
}

// ------------------- Driver -------------------
int main() {
    std::vector<int> sizes = {128, 256, 4096};
    float alpha = 0.5f, beta = 3.0f;

    std::cout << "Alpha = " << alpha << " | Beta = " << beta << std::endl;
    for (int N : sizes) {
        std::cout << "\nMatrix size: " << N << "x" << N << std::endl;

        std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N), h_C_ref(N * N);

        // Fill with random floats
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &x : h_A) x = dist(rng);
        for (auto &x : h_B) x = dist(rng);
        for (auto &x : h_C) x = dist(rng); // initial C
        h_C_ref = h_C; // copy for reference

        // CPU reference
        if (N < 1024) gemm_cpu(h_A.data(), h_B.data(), h_C_ref.data(), N, alpha, beta);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));

        // ---- Benchmark kernel ----
        float ms = benchmark(kernel, d_A, d_B, d_C, N, 10, alpha, beta);
        double perf = gflops(N, ms);

        // ---- Benchmark cublas ----
        float ms_cublas = benchmark(kernel_cublas, d_A, d_B, d_C, N, 10, alpha, beta);
        double perf_cublas = gflops(N, ms_cublas);

        // Copy result back
        CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
        kernel(d_A, d_B, d_C, N, N, N, alpha, beta);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost));

        // Check correctness
        double max_err = 0.0;
        if (N < 1024){
            for (int i = 0; i < N * N; i++) {
                max_err = std::max(max_err, static_cast<double>(std::abs(h_C[i] - h_C_ref[i])));
            }
        }
        if (N < 1024){
            std::cout << "Average Time Elapsed: " << ms << " ms, "
                    << perf << " GFLOPS"
                      << " | Max error: " << max_err 
                    << std::endl;
        } else {
            std::cout << "Average Time Elapsed: " << ms << " ms, "
                    << perf << " GFLOPS" << std::endl;
        }
        std::cout << "Perfomance relative to cuBLAS: " << (perf/perf_cublas) * 100 << "%" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    return 0;
}
