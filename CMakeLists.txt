cmake_minimum_required(VERSION 3.28)
project(CUDA_Practice CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(CUDA_Practice
        src/6_optimized_matmul/main.cu
        src/6_optimized_matmul/Matrix.cu
        src/6_optimized_matmul/CudaUtil.cu
        src/6_optimized_matmul/Timer.cu
        src/6_optimized_matmul/Matrix.cuh
        src/6_optimized_matmul/CudaUtil.cuh
        src/6_optimized_matmul/Timer.cuh
)

set_target_properties(CUDA_Practice PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
