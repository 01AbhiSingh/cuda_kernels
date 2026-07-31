#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void calculateSquares (int* output, int n){
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex <= n){
        output[globalIndex] = globalIndex * globalIndex; 
    }
}

int main(){
    const int n = 8;
    const int bytes = n * sizeof(int);
    std::vector<int> hostOutput(n,0);
    
    int* deviceOutput = nullptr;

    cudaMalloc(&deviceOutput, bytes);
    calculateSquares<<<2,6>>>(deviceOutput,n);
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost);
    for (int value : hostOutput) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    // Release GPU memory
    cudaFree(deviceOutput);
    return 0;
}