#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrixMultiply(
    const int* A,
    const int* B,
    int* C,
    int rowsA,
    int colsA,
    int colsB)
{
    int column =
        blockIdx.x * blockDim.x + threadIdx.x;

    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rowsA && column < colsB) {

        int sum = 0;

        for (int k = 0; k < colsA; ++k) {

            int indexA =
                row * colsA + k;

            int indexB =
                k * colsB + column;

            sum += A[indexA] * B[indexB];
        }

        int indexC =
            row * colsB + column;

        C[indexC] = sum;
    }
}

int main()
{
    const int rowsA = 2;
    const int colsA = 3;

    const int rowsB = 3;
    const int colsB = 2;

    std::vector<int> A = {
        1, 2, 3,
        4, 5, 6
    };

    std::vector<int> B = {
        7,  8,
        9, 10,
        11, 12
    };

    const int elementsA = rowsA * colsA;
    const int elementsB = rowsB * colsB;
    const int elementsC = rowsA * colsB;

    const size_t bytesA =
        elementsA * sizeof(int);

    const size_t bytesB =
        elementsB * sizeof(int);

    const size_t bytesC =
        elementsC * sizeof(int);

    std::vector<int> C(elementsC, 0);

    int* deviceA = nullptr;
    int* deviceB = nullptr;
    int* deviceC = nullptr;

    cudaMalloc(&deviceA, bytesA);
    cudaMalloc(&deviceB, bytesB);
    cudaMalloc(&deviceC, bytesC);

    cudaMemcpy(
        deviceA,
        A.data(),
        bytesA,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        deviceB,
        B.data(),
        bytesB,
        cudaMemcpyHostToDevice
    );

    dim3 threadsPerBlock(2, 2);

    dim3 numberOfBlocks(
        (colsB + threadsPerBlock.x - 1)
            / threadsPerBlock.x,

        (rowsA + threadsPerBlock.y - 1)
            / threadsPerBlock.y
    );

    matrixMultiply<<<numberOfBlocks, threadsPerBlock>>>(
        deviceA,
        deviceB,
        deviceC,
        rowsA,
        colsA,
        colsB
    );

    cudaError_t error =
        cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr
            << "CUDA error: "
            << cudaGetErrorString(error)
            << "\n";

        return 1;
    }

    cudaMemcpy(
        C.data(),
        deviceC,
        bytesC,
        cudaMemcpyDeviceToHost
    );

    std::cout << "Result matrix:\n";

    for (int row = 0; row < rowsA; ++row) {

        for (int column = 0;
             column < colsB;
             ++column) {

            int index =
                row * colsB + column;

            std::cout
                << C[index]
                << "\t";
        }

        std::cout << "\n";
    }

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}