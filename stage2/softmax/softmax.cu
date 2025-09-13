#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <torch/extension.h>
#include <cuda.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

__global__ void softmax_naive(const float *scores, float *result, int N){
    uint i = blockDim.x * blockIdx.x + threadIdx.x;

    float max;
    if(i == 0){
        max = scores[0];
        for(int idx = 0; idx < N; ++idx){
            if(max < scores[idx]) max = scores[idx];
        }
    }
    __syncthreads();
    if (i < N){
        result[i] = __expf(scores[i] - max);
    }
    __syncthreads();
    float sum;
    if(i == 0){
        sum = 0.0f;
        for(int idx = 0; idx < N; ++idx){
            sum += result[idx];
        }
    }
    __syncthreads();
    if(i < N){
        result[i] /= sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int N = input.size(0);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    softmax_naive<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_cuda", &softmax_cuda, "Custom Softmax CUDA");
}

// int main(){
//     int N = 3;
//     size_t size = N * sizeof(float);
//     std::vector<float> h_scores = {2.0, 1.0, 0.1};
//     std::vector<float> h_res(N);
//     float *d_scores, *d_res;

//     CUDA_CHECK(cudaMalloc(&d_scores, size));
//     CUDA_CHECK(cudaMalloc(&d_res, size));

//     CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), size, cudaMemcpyHostToDevice));

//     dim3 block(256, 1, 1);
//     dim3 grid(( N + block.x - 1) / block.x, 1, 1);

//     softmax_naive<<<grid, block>>>(d_scores, d_res, N);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, size, cudaMemcpyDeviceToHost));

//     std::cout << "Result: [ ";
//     for(int i = 0; i < N; ++i){
//         std::cout << h_res[i] << " ";
//     }
//     std::cout << "]" << std::endl;

//     CUDA_CHECK(cudaFree(d_scores));  CUDA_CHECK(cudaFree(d_res));
// }