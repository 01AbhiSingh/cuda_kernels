#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void reduceSum_2(const int* input, int* output, int n){
    extern __shared__ int sharedData[];
    int localIndex = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex < n){
        sharedData[localIndex] = input[globalIndex];
    }

    else{
        sharedData[localIndex] = 0;
    }
    __syncthreads();

    for(int stride = blockDim.x/2;stride>0;stride/=2){
        if(localIndex<stride){
            sharedData[localIndex] += sharedData[localIndex + stride];
        }
        __syncthreads();
    }
    if(localIndex == 0){
        output[blockIdx.x] = sharedData[0];
    }
}

int main(){
    std::vector<int> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    const int n = static_cast<int>(input.size());

    const size_t bytes = n * sizeof(int);
    const int threadsPerBlock = 8;

    const int numberOfBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<int> partialSums(numberOfBlocks,0);

    int* deviceInput = nullptr;
    int* deviceOutput =nullptr;

    cudaMalloc(&deviceInput, bytes);
    cudaMalloc(&deviceOutput, numberOfBlocks * sizeof(int));

    cudaMemcpy(deviceInput,input.data(),bytes,cudaMemcpyHostToDevice);

    const size_t sharedMemoryBytes = threadsPerBlock * sizeof(int);

    reduceSum_2<<<numberOfBlocks,threadsPerBlock,sharedMemoryBytes>>>(deviceInput,deviceOutput,n);

    cudaError_t error = cudaDeviceSynchronize();

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