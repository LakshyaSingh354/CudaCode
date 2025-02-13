#include<iostream>
using namespace std;

void mat_mul(float* a, float* b, float* c, int m, int k, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            float sum = 0;
            for(int l = 0; l < k; l++){
                sum += a[i*k + l] * b[l*n + j];
            }
            c[i*n + j] = sum;
        }
    }
}

int main(){
    int m = 2, k = 3, n = 2;
    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {1, 2, 3, 4, 5, 6};
    float c[m*n];
    mat_mul(a, b, c, m, k, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            cout << c[i*n + j] << " ";
        }
        cout << endl;
    }
    return 0;
}