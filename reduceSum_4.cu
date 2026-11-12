#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void reduceSum(const int* input,int* output,int n){
    // Shared memory is private to each block.
    // Each block gets its own sharedData array.
    extern __shared__ int sharedData[];

    // Thread number inside the current block
    int localIndex = threadIdx.x;

    // Unique thread index across the whole grid
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one value from global memory
    // into shared memory.
    if (globalIndex < n)sharedData[localIndex] = input[globalIndex];

    else {
        sharedData[localIndex] = 0;
    }

    // Wait until all threads in THIS block
    // have finished loading their values.
    __syncthreads();

    // Parallel reduction inside each block.
    //
    // For 8 threads:
    // stride = 4
    // stride = 2
    // stride = 1
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (localIndex < stride) {
            sharedData[localIndex] +=
                sharedData[localIndex + stride];
        }

        // Wait before starting the next reduction step
        __syncthreads();
    }

    // After the reduction,
    // sharedData[0] contains the sum for THIS block.
    //
    // Only thread 0 writes the block result.
    if (localIndex == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main()
{
    std::vector<int> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    const int n =
        static_cast<int>(input.size());

    const size_t inputBytes =
        n * sizeof(int);

    // We want 8 threads in each block
    const int threadsPerBlock = 8;

    // Calculate number of blocks
    const int numberOfBlocks =
        (n + threadsPerBlock - 1)
        / threadsPerBlock;

    // Here:
    //
    // n = 16
    // threadsPerBlock = 8
    //
    // numberOfBlocks = 2

    // Each block produces ONE partial sum.
    //
    // Therefore we need space for 2 results.
    std::vector<int> partialSums(
        numberOfBlocks,
        0
    );

    int* deviceInput = nullptr;
    int* deviceOutput = nullptr;

    // Allocate GPU memory for input
    cudaMalloc(
        &deviceInput,
        inputBytes
    );

    // Allocate GPU memory for partial sums
    cudaMalloc(
        &deviceOutput,
        numberOfBlocks * sizeof(int)
    );

    // Copy input from CPU to GPU
    cudaMemcpy(
        deviceInput,
        input.data(),
        inputBytes,
        cudaMemcpyHostToDevice
    );

    // Each block needs shared memory for 8 integers
    const size_t sharedMemoryBytes =
        threadsPerBlock * sizeof(int);

    // Launch:
    //
    // 2 blocks
    // 8 threads per block
    // shared memory for 8 integers per block
    reduceSum<<<
        numberOfBlocks,
        threadsPerBlock,
        sharedMemoryBytes>>>(
            deviceInput,
            deviceOutput,
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

    // Copy the block-level partial sums
    // from GPU back to CPU
    cudaMemcpy(
        partialSums.data(),
        deviceOutput,
        numberOfBlocks * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    std::cout << "Partial sums:\n";

    for (int value : partialSums) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    // Final reduction done on CPU
    int finalSum = 0;

    for (int value : partialSums) {
        finalSum += value;
    }

    std::cout
        << "Final sum = "
        << finalSum
        << "\n";

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return 0;
}