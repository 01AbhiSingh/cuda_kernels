#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_gpu(){
    printf("hello from gpu\n");
}

int main(){
    printf("hello from cpu\n");
    hello_gpu<<<1,1>>>();

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess){
            printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    return 0;
}