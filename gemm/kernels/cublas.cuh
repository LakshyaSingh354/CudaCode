#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// --- Helper to check for cuBLAS errors ---
// This is a good practice, similar to our CUDA_CHECK macro.
static const char* _cublasGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << _cublasGetErrorEnum(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }


// --- cuBLAS Handle Manager ---
// This is a neat trick to avoid the performance cost of creating and destroying
// the cuBLAS handle every single time we call the function, especially in a benchmark loop.
// It creates one handle and safely cleans it up when the program exits.
class CublasHandle {
public:
    static cublasHandle_t get() {
        static CublasHandle instance;
        return instance.handle;
    }

private:
    cublasHandle_t handle;
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    ~CublasHandle() {
        cublasDestroy(handle);
    }
    CublasHandle(const CublasHandle&) = delete;
    void operator=(const CublasHandle&) = delete;
};


// --- The cuBLAS SGEMM Wrapper Function ---

void sgemm_cublas_wrapper(const float* A, const float* B, float* C, 
                          int M, int K, int N, 
                          float alpha, float beta) {

    cublasHandle_t handle = CublasHandle::get();
    
    // IMPORTANT NOTE on Row-Major vs. Column-Major Layouts:
    // our code uses a "row-major" layout for matrices, which is standard.
    // However, cuBLAS (and FORTRAN/BLAS libraries in general) expect a "column-major" layout.
    // To calculate C = A * B with row-major matrices, we can use a mathematical trick:
    // (A * B)^T = B^T * A^T
    // If we treat our row-major matrices as column-major ones, their dimensions and data are effectively transposed.
    // So, we can ask cuBLAS to calculate `C_col = B_col * A_col`, which gives us the correct result
    // without needing to actually transpose any data in memory.
    // This means we swap the A and B pointers and also swap the M and N dimensions in the call.
    
    // Parameters for cuBLAS call (swapped M and N)
    int cublas_m = N;
    int cublas_n = M;
    int cublas_k = K;

    // Leading dimensions
    int lda = N; // Leading dimension of B in row-major is N
    int ldb = K; // Leading dimension of A in row-major is K
    int ldc = N; // Leading dimension of C in row-major is N

    CUBLAS_CHECK(cublasSgemm(handle, 
                             CUBLAS_OP_N, CUBLAS_OP_N, 
                             cublas_m, cublas_n, cublas_k, 
                             &alpha, 
                             B, lda,  // Pass B as the first matrix
                             A, ldb,  // Pass A as the second matrix
                             &beta, 
                             C, ldc));
}

// --- Launcher Function ---

void kernel_cublas(const float* A, const float* B, float* C,
                   int M, int K, int N, 
                   float alpha, float beta) {
    sgemm_cublas_wrapper(A, B, C, M, K, N, alpha, beta);
}

// void kernel(const float* A, const float* B, float* C,
//                    int M, int K, int N, 
//                    float alpha, float beta) {
//     sgemm_cublas_wrapper(A, B, C, M, K, N, alpha, beta);
// }