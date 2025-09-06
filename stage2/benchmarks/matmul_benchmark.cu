#include <iostream>
#include <vector>
#include <random>
#include <iomanip>


#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ \
                  << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
#define TILE_SIZE 32

void matmul_naive_cpu(float* A, float* B, float* C, int m, int k, int n){
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            float sum = 0.0f;
            for(int p = 0; p < k; ++p){
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

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

        for(int p = 0; p < TILE_SIZE; p++){
            sum += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        }
        
        __syncthreads();
    }

    if(row < m && col < n){
        C[row * n + col] = sum;
    }
}

void verify_result(const float* C_gpu, int m, int n, const float* A, const float* B, int k) {
    std::cout << "Verifying result on the CPU..." << std::endl;
    std::vector<float> C_cpu(m * n);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[row * k + p] * B[p * n + col];
            }
            C_cpu[row * n + col] = sum;
        }
    }

    float max_error = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        max_error = fmax(max_error, fabs(C_gpu[i] - C_cpu[i]));
    }
    
    // A small tolerance for floating point inaccuracies
    float tolerance = 1e-4;
    if (max_error <= tolerance) {
        std::cout << "✅ Verification PASSED! Max error: " << max_error << std::endl;
    } else {
        std::cout << "❌ Verification FAILED! Max error: " << max_error << std::endl;
    }
}


int main() {
    int m = 2048;
    int k = 1024;
    int n = 2048;

    int warmup_runs = 5;
    int timed_runs = 20;

    std::cout << "Starting Matrix Multiplication Benchmark" << std::endl;
    std::cout << "Matrix Dimensions: " << m << "x" << k << " * " << k << "x" << n << std::endl;
    std::cout << "TILE_SIZE: " << TILE_SIZE << std::endl;
    std::cout << "Warmup Runs: " << warmup_runs << " | Timed Runs: " << timed_runs << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    size_t A_size = m * k * sizeof(float);
    size_t B_size = k * n * sizeof(float);
    size_t C_size = m * n * sizeof(float);

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_C(m * n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < m * k; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < k * n; ++i) h_B[i] = dis(gen);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, A_size));
    CUDA_CHECK(cudaMalloc(&d_B, B_size));
    CUDA_CHECK(cudaMalloc(&d_C, C_size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_size, cudaMemcpyHostToDevice));

    // --- Benchmarking Setup ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total_time = 0.0f;

    // =======================================================
    // Naive Kernel Benchmark
    // =======================================================
    std::cout << "\nBenchmarking NAIVE kernel..." << std::endl;
    
    dim3 threads_per_block_naive(16, 16);
    dim3 num_blocks_naive( (n + threads_per_block_naive.x - 1) / threads_per_block_naive.x,
                           (m + threads_per_block_naive.y - 1) / threads_per_block_naive.y );

    for (int i = 0; i < warmup_runs; ++i) {
        matmul_naive<<<num_blocks_naive, threads_per_block_naive>>>(d_A, d_B, d_C, m, k, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    total_time = 0.0f;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; ++i) {
        matmul_naive<<<num_blocks_naive, threads_per_block_naive>>>(d_A, d_B, d_C, m, k, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));

    float avg_time_naive = total_time / timed_runs;
    std::cout << "Average Naive Kernel Time: " << std::fixed << std::setprecision(3) << avg_time_naive << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, C_size, cudaMemcpyDeviceToHost));
    verify_result(h_C.data(), m, n, h_A.data(), h_B.data(), k);


    // =======================================================
    // Tiled Kernel Benchmark
    // =======================================================
    std::cout << "\nBenchmarking TILED kernel..." << std::endl;

    dim3 threads_per_block_tiled(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks_tiled( (n + TILE_SIZE - 1) / TILE_SIZE,
                           (m + TILE_SIZE - 1) / TILE_SIZE );

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        matmul_tiled<<<num_blocks_tiled, threads_per_block_tiled>>>(d_A, d_B, d_C, m, k, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    total_time = 0.0f;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; ++i) {
        matmul_tiled<<<num_blocks_tiled, threads_per_block_tiled>>>(d_A, d_B, d_C, m, k, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));
    
    float avg_time_tiled = total_time / timed_runs;
    std::cout << "Average Tiled Kernel Time: " << std::fixed << std::setprecision(3) << avg_time_tiled << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, C_size, cudaMemcpyDeviceToHost));
    verify_result(h_C.data(), m, n, h_A.data(), h_B.data(), k);

    // =======================================================
    // Final Results
    // =======================================================
    std::cout << "\n--- Final Score ---" << std::endl;
    std::cout << std::left << std::setw(15) << "Naive Kernel:" << std::right << std::setw(10) << avg_time_naive << " ms" << std::endl;
    std::cout << std::left << std::setw(15) << "Tiled Kernel:" << std::right << std::setw(10) << avg_time_tiled << " ms" << std::endl;
    if (avg_time_tiled > 0) {
        std::cout << "\nSpeedup: The tiled kernel is " << std::fixed << std::setprecision(2) 
                  << avg_time_naive / avg_time_tiled << "x faster! 🚀" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}