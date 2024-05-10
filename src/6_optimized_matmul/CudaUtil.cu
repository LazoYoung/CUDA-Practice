#include "CudaUtil.cuh"

__host__ void printDeviceInfo() {
    int device;
    cudaDeviceProp prop {};
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("CUDA device info:\n");
    printf("# of registers per block: %d\n", prop.regsPerBlock);
    printf("# of threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("# of threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Shared memory per block: %zu\n", prop.sharedMemPerBlock - prop.reservedSharedMemPerBlock);
    printf("Shared memory per SM: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("\n");
}
