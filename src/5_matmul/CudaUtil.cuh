#ifndef CUDAUTIL_CUH
#define CUDAUTIL_CUH

#include <iostream>
#include <string>

#define cudaCheckError()                                                         \
auto error = cudaGetLastError();                                                 \
if (error != cudaSuccess) {                                                      \
    printf("%s from %s:%d\n", cudaGetErrorName(error), __FILE__, __LINE__);      \
    printf("%s\n", cudaGetErrorString(error));                                   \
}                                                                                \

__host__ void printDeviceInfo();

#endif // CUDAUTIL_CUH
