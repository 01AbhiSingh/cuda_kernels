#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void coalescedCopy(const float* input,float* output,int n){
    int index =blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        output[index] = input[index];
    }
}

__global__ void stridedCopy(const float* input,float* output,int rows,int columns){
    // Important:
    // threadIdx.x changes fastest inside a warp.
    //
    // Here x is mapped to ROW.
    // So neighboring threads access different rows.
    int row =
        blockIdx.x * blockDim.x + threadIdx.x;

    int column =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && column < columns) {

        int index =
            row * columns + column;

        output[index] = input[index];
    }
}

int main()
{
    const int rows = 4096;
    const int columns = 4096;

    const int n =
        rows * columns;

    const size_t bytes =
        static_cast<size_t>(n) * sizeof(float);

    // -----------------------------------------
    // CPU data
    // -----------------------------------------

    std::vector<float> input(n, 1.0f);

    std::vector<float> outputCoalesced(n, 0.0f);
    std::vector<float> outputStrided(n, 0.0f);

    // -----------------------------------------
    // GPU pointers
    // -----------------------------------------

    float* deviceInput = nullptr;
    float* deviceOutputCoalesced = nullptr;
    float* deviceOutputStrided = nullptr;

    cudaMalloc(&deviceInput, bytes);

    cudaMalloc(
        &deviceOutputCoalesced,
        bytes);

    cudaMalloc(
        &deviceOutputStrided,
        bytes);

    // Copy input CPU -> GPU
    cudaMemcpy(
        deviceInput,
        input.data(),
        bytes,
        cudaMemcpyHostToDevice);

    // =========================================
    // COALESCED VERSION
    // =========================================

    const int threadsPerBlock = 256;

    const int numberOfBlocks =
        (n + threadsPerBlock - 1)
        / threadsPerBlock;

    cudaEvent_t startCoalesced;
    cudaEvent_t stopCoalesced;

    cudaEventCreate(&startCoalesced);
    cudaEventCreate(&stopCoalesced);

    // Record start time
    cudaEventRecord(startCoalesced);

    coalescedCopy<<<
        numberOfBlocks,
        threadsPerBlock>>>(
            deviceInput,
            deviceOutputCoalesced,
            n);

    // Record stop event
    cudaEventRecord(stopCoalesced);

    // Wait for kernel to finish
    cudaEventSynchronize(stopCoalesced);

    float coalescedTime = 0.0f;

    cudaEventElapsedTime(
        &coalescedTime,
        startCoalesced,
        stopCoalesced);

    // =========================================
    // STRIDED VERSION
    // =========================================

    dim3 threadsPerBlock2D(32, 8);

    dim3 numberOfBlocks2D(
        (rows + threadsPerBlock2D.x - 1)
            / threadsPerBlock2D.x,

        (columns + threadsPerBlock2D.y - 1)
            / threadsPerBlock2D.y
    );

    cudaEvent_t startStrided;
    cudaEvent_t stopStrided;

    cudaEventCreate(&startStrided);
    cudaEventCreate(&stopStrided);

    cudaEventRecord(startStrided);

    stridedCopy<<<
        numberOfBlocks2D,
        threadsPerBlock2D>>>(
            deviceInput,
            deviceOutputStrided,
            rows,
            columns);

    cudaEventRecord(stopStrided);

    cudaEventSynchronize(stopStrided);

    float stridedTime = 0.0f;

    cudaEventElapsedTime(
        &stridedTime,
        startStrided,
        stopStrided);

    cudaMemcpy(
        outputCoalesced.data(),
        deviceOutputCoalesced,
        bytes,
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        outputStrided.data(),
        deviceOutputStrided,
        bytes,
        cudaMemcpyDeviceToHost);



    bool correct = true;

    for (int i = 0; i < n; ++i) {

        if (std::abs(
                outputCoalesced[i]
                - outputStrided[i]) > 1e-6f)
        {
            correct = false;

            std::cout
                << "Mismatch at index "
                << i
                << "\n";

            break;
        }
    }

    std::cout
        << "Coalesced kernel time: "
        << coalescedTime
        << " ms\n";

    std::cout
        << "Strided kernel time: "
        << stridedTime
        << " ms\n";

    if (correct) {
        std::cout
            << "Verification passed\n";
    }
    else {
        std::cout
            << "Verification failed\n";
    }


    cudaEventDestroy(startCoalesced);
    cudaEventDestroy(stopCoalesced);

    cudaEventDestroy(startStrided);
    cudaEventDestroy(stopStrided);

    cudaFree(deviceInput);
    cudaFree(deviceOutputCoalesced);
    cudaFree(deviceOutputStrided);

    return 0;
}