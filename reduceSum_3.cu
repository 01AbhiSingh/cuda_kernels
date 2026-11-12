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

int main (){
    std::vector<int> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    return 0;
}