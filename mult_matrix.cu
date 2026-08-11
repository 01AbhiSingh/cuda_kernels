#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrix_mult(const int* a, int* output, int scalar,int rows, int columns){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < rows && column < columns){
        int index = row * columns + column;
        output[index] = scalar * a[index];
    }
}

int main (){
    const int rows = 3;
    const int columns = 4;
    const int totalElements = rows * columns;
    const int scalar  = 4;
    const size_t bytes = totalElements * sizeof(int);

    std::vector<int> A = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    };

    std::vector<int> C(totalElements, 0);

    int* deviceA, *deviceC = nullptr;


    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceC, bytes);
    cudaMemcpy(deviceA,A.data(),bytes,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(2,2);
    dim3 numberOfBlocks(
        (columns + threadsPerBlock.x - 1)
            / threadsPerBlock.x,

        (rows + threadsPerBlock.y - 1)
            / threadsPerBlock.y
    );

    matrix_mult<<<numberOfBlocks, threadsPerBlock>>>(deviceA, deviceC,scalar, rows,columns);
    cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
        std::cerr
            << "CUDA error: "
            << cudaGetErrorString(error)
            << "\n";

        return 1;
    }

    cudaMemcpy(C.data(), deviceC,bytes,cudaMemcpyDeviceToHost);

    std::cout << "Result matrix:\n";

    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            int index = row * columns + column;

            std::cout << C[index] << "\t";
        }

        std::cout << "\n";
    }

    cudaFree(deviceA);
    cudaFree(deviceC);

    return 0;

}