#include <cuda_runtime.h>
#include <cstdio>

__global__ void showThreadInfo()
{
    int globalIndex =
        blockIdx.x * blockDim.x + threadIdx.x;

    printf(
        "Block: %d, Thread: %d, Global index: %d\n",
        blockIdx.x,
        threadIdx.x,
        globalIndex
    );
}

int main()
{
    printf("Program started on CPU\n");

    showThreadInfo<<<2, 4>>>();

    cudaError_t error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        printf(
            "CUDA error: %s\n",
            cudaGetErrorString(error)
        );

        return 1;
    }

    printf("GPU work completed\n");

    return 0;
}   