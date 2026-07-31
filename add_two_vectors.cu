#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void addVectors(
    const int* a,
    const int* b,
    int* c,
    int n)
{
    int globalIndex =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex < n) {
        c[globalIndex] =
            a[globalIndex] + b[globalIndex];
    }
}

int main()
{
    std::vector<int> A =
        {1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<int> B =
        {10, 20, 30, 40, 50, 60, 70, 80};

    const int n = static_cast<int>(A.size());
    const size_t bytes = n * sizeof(int);

    std::vector<int> C(n, 0);

    int* deviceA = nullptr;
    int* deviceB = nullptr;
    int* deviceC = nullptr;

    // Allocate GPU memory
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);

    // Copy input vectors from CPU to GPU
    cudaMemcpy(
        deviceA,
        A.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        deviceB,
        B.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    // Launch 2 blocks with 4 threads each
    addVectors<<<2, 4>>>(
        deviceA,
        deviceB,
        deviceC,
        n
    );

    cudaDeviceSynchronize();

    // Copy output from GPU to CPU
    cudaMemcpy(
        C.data(),
        deviceC,
        bytes,
        cudaMemcpyDeviceToHost
    );

    // Print output
    for (int value : C) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    // Release GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}