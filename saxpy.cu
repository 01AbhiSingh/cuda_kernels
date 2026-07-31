#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void saxpy(
    const float* a,
    const float* b,
    float* c,
    float alpha,
    int n)
{
    int globalIndex =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex < n) {
        c[globalIndex] =
            alpha * a[globalIndex] + b[globalIndex];
    }
}

int main()
{
    std::vector<float> a = {2, 45, 75, 32};
    std::vector<float> b = {34, 766, 23, 56};

    const int n = static_cast<int>(a.size());
    const size_t bytes = n * sizeof(float);
    const float alpha = 8.0f;

    std::vector<float> c(n, 0.0f);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);

    cudaMemcpy(
        deviceA,
        a.data(),
        bytes,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        deviceB,
        b.data(),
        bytes,
        cudaMemcpyHostToDevice);

    const int threadsPerBlock = 4;
    const int blocks =
        (n + threadsPerBlock - 1) / threadsPerBlock;

    saxpy<<<blocks, threadsPerBlock>>>(
        deviceA,
        deviceB,
        deviceC,
        alpha,
        n);

    cudaError_t error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr
            << "CUDA error: "
            << cudaGetErrorString(error)
            << "\n";
    }

    cudaMemcpy(
        c.data(),
        deviceC,
        bytes,
        cudaMemcpyDeviceToHost);

    for (float value : c) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}