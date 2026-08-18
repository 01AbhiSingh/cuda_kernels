#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void reduceSum(const int* input, int* output, int n){
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

    for (int stride = blockDim.x/2;stride>0;stride/=2){
        if (localIndex < stride){
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
        1, 2, 3, 4, 5, 6, 7, 8
    };

    const int n = static_cast<int>(input.size());

    const size_t inputBytes = n * sizeof(int);

    std::vector<int> result(1, 0);

    int* deviceInput = nullptr;
    int* deviceOutput = nullptr;

    cudaMalloc(
        &deviceInput,
        inputBytes);

    cudaMalloc(
        &deviceOutput,
        sizeof(int));

    cudaMemcpy(
        deviceInput,
        input.data(),
        inputBytes,
        cudaMemcpyHostToDevice);

    // Since there are exactly 8 input values,
    // we use exactly 8 threads.
    const int threadsPerBlock = 8;

    const size_t sharedMemoryBytes =
        threadsPerBlock * sizeof(int);

    // Kernel launch:
    //
    // <<<1, threadsPerBlock, sharedMemoryBytes>>>
    //
    // First argument:
    //     1 block
    //
    // Second argument:
    //     8 threads inside that block
    //
    // Third argument:
    //     amount of dynamic shared memory
    reduceSum<<<1,threadsPerBlock,sharedMemoryBytes>>>(deviceInput,deviceOutput,n);

    // Wait until the GPU finishes.
    //
    // Also allows us to detect runtime kernel errors.
    cudaError_t error =
        cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr
            << "CUDA error: "
            << cudaGetErrorString(error)
            << "\n";

        return 1;
    }

    // Copy final sum from GPU memory
    // back to CPU memory.
    cudaMemcpy(
        result.data(),
        deviceOutput,
        sizeof(int),
        cudaMemcpyDeviceToHost);

    // Print final result.
    std::cout
        << "Sum = "
        << result[0]
        << "\n";

    // Release GPU memory.
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return 0;
}