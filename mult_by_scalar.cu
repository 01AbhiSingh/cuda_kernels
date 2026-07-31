#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void scalarMultiply(const int* v, int* b, int scalar, int n ){
    int global_index = blockIdx.x * blockDim.x +threadIdx.x;
    if( global_index<n){
        b[global_index] = scalar * v[global_index];
    }
}

int main(){

    const int scalar = 4;
    std::vector<int>v = {2,4,6,4,2};
    const int n = static_cast<int>(v.size());
    int bytes = n * sizeof(int);

    std::vector<int>b(n,0);
    int* deviceV = nullptr;
    int* deviceB = nullptr;

    cudaMalloc(&deviceV, bytes);
    cudaMalloc(&deviceB, bytes);

    cudaMemcpy(deviceV,v.data(),bytes,cudaMemcpyHostToDevice);
    scalarMultiply<<<2,4>>>(deviceV,deviceB,scalar,n);
    cudaDeviceSynchronize();
    cudaMemcpy(b.data(), deviceB, bytes, cudaMemcpyDeviceToHost);

    for (auto it : b){
        std::cout<< it << " " << std::endl;
    }
    cudaFree(deviceV);
    cudaFree(deviceB);
    return 0;  
}