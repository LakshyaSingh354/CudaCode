#include <iostream>
#include <chrono>

int main() {
    const size_t N = 1000000000;  // 1 billion elements

    // Allocate memory dynamically on heap
    int* vector_a = new int[N];
    int* vector_b = new int[N];
    int* vector_c = new int[N];

    // Initialize vectors
    for(size_t i = 0; i < N; i++) {
        vector_a[i] = i;
        vector_b[i] = N - i;
    }

    auto start = std::chrono::high_resolution_clock::now();  // Start timing

    // Perform vector addition
    for(size_t i = 0; i < N; i++) {
        vector_c[i] = vector_a[i] + vector_b[i];
    }

    auto stop = std::chrono::high_resolution_clock::now();  // Stop timing
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    int result_sum = 0;
    for(size_t i = 0; i < N; i++) {
        result_sum += vector_c[i];
    }
    
    std::cout << "CPU Result: sum = " << result_sum << ", Time = " << duration.count() << " us\n";

    // Free memory
    delete[] vector_a;
    delete[] vector_b;
    delete[] vector_c;

    return 0;
}