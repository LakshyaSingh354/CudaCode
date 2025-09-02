#include <iostream>
using namespace std;

void matmul_naive(float* A, float* B, float* C, int m, int k, int n){
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

int main(){
    int m = 5, k = 4, n = 9;

    float A[m*k], B[k*n], C[m*n];

    for (int i = 0; i < m; ++i){
        for(int j = 0; j < k; ++j){
            A[i*k + j] = i*0.1 + j*0.2;
        }
    }
    for (int i = 0; i < k; ++i){
        for(int j = 0; j < n; ++j){
            B[i*n + j] = i*0.2 + j*0.1;
        }
    }
    cout << "A = [";
    for (int i = 0; i < m; ++i){
        cout << "[ ";
        for(int j = 0; j < k; ++j){
            cout << A[i*k + j] << " ";
        }
        cout << "]\n";
    }

    cout << "]" << endl;
    cout << "B = [";
    for (int i = 0; i < m; ++i){
        cout << "[ ";
        for(int j = 0; j < k; ++j){
            cout << B[i*n + j] << " ";
        }
        cout << "]\n";
    }
    cout << "]" << endl;

    matmul_naive(A, B, C, m, k, n);

    cout << "C = [";
    for (int i = 0; i < m; ++i){
        cout << "[ ";
        for(int j = 0; j < k; ++j){
            cout << C[i*n + j] << " ";
        }
        cout << "]\n";
    }
    cout << "]" << endl;
}