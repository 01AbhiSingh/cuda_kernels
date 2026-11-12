#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void reduceSum(
    const int* input,
    int* output,
    int n)
{
    // Shared memory belongs to ONE block.
    // Every thread in the block can access it.
    extern __shared__ int sharedData[];

    // Thread index inside the block
    int localIndex = threadIdx.x;

    // Global thread index across the grid
    int globalIndex =
        blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element from global memory
    // into shared memory.
    if (globalIndex < n) {
        sharedData[localIndex] =
            input[globalIndex];
    }
    else {
        // Extra threads contribute zero.
        sharedData[localIndex] = 0;
    }

    // Wait until all threads have finished loading.
    __syncthreads();

    // Parallel reduction inside each block.
    //
    // Example with 8 threads:
    //
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

        // Make sure this reduction stage is complete
        // before starting the next one.
        __syncthreads();
    }

    // Thread 0 contains the final sum for this block.
    // Each block writes one partial sum.
    if (localIndex == 0) {
        output[blockIdx.x] =
            sharedData[0];
    }
}

int main()
{
    std::vector<int> input = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32
    };

    const int n =
        static_cast<int>(input.size());

    const size_t inputBytes =
        n * sizeof(int);

    // --------------------------------------------------
    // FIRST REDUCTION STAGE
    // --------------------------------------------------

    const int threadsPerBlock = 8;

    const int numberOfBlocks =
        (n + threadsPerBlock - 1)
        / threadsPerBlock;

    // n = 32
    // threadsPerBlock = 8
    //
    // numberOfBlocks = 4

    int* deviceInput = nullptr;
    int* devicePartialSums = nullptr;

    cudaMalloc(
        &deviceInput,
        inputBytes);

    cudaMalloc(
        &devicePartialSums,
        numberOfBlocks * sizeof(int));

    // Copy original input to GPU.
    cudaMemcpy(
        deviceInput,
        input.data(),
        inputBytes,
        cudaMemcpyHostToDevice);

    // Each block gets shared memory for 8 ints.
    const size_t sharedMemoryBytes =
        threadsPerBlock * sizeof(int);

    // First reduction:
    //
    // 32 input values
    //      ↓
    // 4 blocks
    //      ↓
    // 4 partial sums
    reduceSum<<<
        numberOfBlocks,
        threadsPerBlock,
        sharedMemoryBytes>>>(
            deviceInput,
            devicePartialSums,
            n);

    cudaError_t error =
        cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr
            << "CUDA error after first reduction: "
            << cudaGetErrorString(error)
            << "\n";

        return 1;
    }

    // Optional: copy partial sums back
    // just so we can inspect them.
    std::vector<int> partialSums(
        numberOfBlocks,
        0);

    cudaMemcpy(
        partialSums.data(),
        devicePartialSums,
        numberOfBlocks * sizeof(int),
        cudaMemcpyDeviceToHost);

    std::cout << "First-stage partial sums:\n";

    for (int value : partialSums) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    // --------------------------------------------------
    // SECOND REDUCTION STAGE
    // --------------------------------------------------

    // First stage produced 4 values:
    //
    // 36 100 164 228
    //
    // Now reduce those 4 values on GPU too.

    int* deviceFinalSum = nullptr;

    cudaMalloc(
        &deviceFinalSum,
        sizeof(int));

    const int secondStageThreads =
        numberOfBlocks;   // 4 threads

    const size_t secondStageSharedBytes =
        secondStageThreads * sizeof(int);

    // Second reduction:
    //
    // 4 partial sums
    //      ↓
    // 1 block with 4 threads
    //      ↓
    // 1 final result
    reduceSum<<<
        1,
        secondStageThreads,
        secondStageSharedBytes>>>(
            devicePartialSums,
            deviceFinalSum,
            numberOfBlocks);

    error =
        cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr
            << "CUDA error after second reduction: "
            << cudaGetErrorString(error)
            << "\n";

        return 1;
    }

    // --------------------------------------------------
    // COPY ONLY FINAL RESULT BACK TO CPU
    // --------------------------------------------------

    int finalSum = 0;

    cudaMemcpy(
        &finalSum,
        deviceFinalSum,
        sizeof(int),
        cudaMemcpyDeviceToHost);

    std::cout
        << "Final sum = "
        << finalSum
        << "\n";

    // --------------------------------------------------
    // CLEANUP
    // --------------------------------------------------

    cudaFree(deviceInput);
    cudaFree(devicePartialSums);
    cudaFree(deviceFinalSum);

    return 0;
}