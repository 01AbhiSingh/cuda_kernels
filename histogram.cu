#include<cuda_runtime.h>
#include<iostream>
#include<vector>


__global__ void histogram(const int* input, int* bins, int n){
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex<n){
        int value = input[globalIndex];
        atomicAdd(&bins[value], 1);
    }
}

int main(){
    std::vector<int> input = {
        0, 1, 2, 3,
        1, 2, 2, 3,
        0, 1, 3, 3
    };

    const int n =static_cast<int>(input.size());

    const size_t bytes = n * sizeof(int);

    int* deviceInput = nullptr;
    int* deviceBins  = nullptr;

    // Allocate GPU memory for input vector
    cudaMalloc(&deviceInput, bytes);

    // Allocate GPU memory for a single integer counter
    cudaMalloc(&deviceBins, 4*sizeof(int));

    // Copy input vector from CPU to GPU
    cudaMemcpy(deviceInput,input.data(),bytes,cudaMemcpyHostToDevice);

    // IMPORTANT:
    // deviceCount contains uninitialized GPU memory.
    // Set it to zero before the kernel runs.
    cudaMemset(deviceBins,0,4*  sizeof(int));

    const int threadsPerBlock = 4;

    const int numberOfBlocks =(n + threadsPerBlock - 1) / threadsPerBlock;

    histogram<<<numberOfBlocks, threadsPerBlock>>>(deviceInput,deviceBins,n);

    cudaError_t error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr
            << "CUDA error: "
            << cudaGetErrorString(error)
            << "\n";

        return 1;
    }
    std::vector<int> bins(4, 0);

    // Copy final counter from GPU back to CPU
    cudaMemcpy(bins.data(),deviceBins,4 * sizeof(int),cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; ++i) {
        std::cout << i << " -> " << bins[i] << "\n";
    }

    cudaFree(deviceInput);
    cudaFree(deviceBins);

    return 0;
}