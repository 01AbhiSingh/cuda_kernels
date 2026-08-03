#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void addMatrix(const int* a, const int*b, int* c, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols){
        c[row * cols + col] = a[row * cols + col] + b[row * cols + col];
    }
}

int main(){
    const int rows = 3;
    const int cols = 4;
    const int totalElements = rows * columns;
    const size_t bytes = totalElements * sizeof(int);

    std::vector<int> A = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    };

    std::vector<int> B = {
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120
    };

    std::vector<int> C(rows * cols);
    int *d_A, *d_B, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A,A.data(),bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B.data(),bytes, cudaMemcpyHostToDevice);
    return 0 ;
}