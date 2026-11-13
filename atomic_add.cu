#include<cuda_runtime.h>
#include<iostream>
#include<vector>

__global__ void countEven(const int* input, int* count, int n){
    int globalIndex  = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalIndex<n){
        if(input[globalIndex]%2==0){
            atomicAdd(count,1);
        }

    }
}
int main()
{
    std::vector<int> input = {
        3, 8, 4, 7,
        10, 13, 16, 21,
        24, 30, 33, 40
    };

    const int n =
        static_cast<int>(input.size());

    const size_t bytes =
        n * sizeof(int);

    int* deviceInput = nullptr;
    int* deviceCount = nullptr;

    // Allocate GPU memory for input vector
    cudaMalloc(&deviceInput, bytes);

    // Allocate GPU memory for a single integer counter
    cudaMalloc(&deviceCount, sizeof(int));

    // Copy input vector from CPU to GPU
    cudaMemcpy(
        deviceInput,
        input.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    // IMPORTANT:
    // deviceCount contains uninitialized GPU memory.
    // Set it to zero before the kernel runs.
    cudaMemset(
        deviceCount,
        0,
        sizeof(int)
    );

    const int threadsPerBlock = 4;

    const int numberOfBlocks =
        (n + threadsPerBlock - 1)
        / threadsPerBlock;

    countEven<<<numberOfBlocks, threadsPerBlock>>>(
        deviceInput,
        deviceCount,
        n
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

    int count = 0;

    // Copy final counter from GPU back to CPU
    cudaMemcpy(
        &count,
        deviceCount,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    std::cout
        << "Even numbers = "
        << count
        << "\n";

    cudaFree(deviceInput);
    cudaFree(deviceCount);

    return 0;
}