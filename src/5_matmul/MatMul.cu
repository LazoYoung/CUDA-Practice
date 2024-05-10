#include <cassert>
#include "CudaUtil.cuh"
#include "Matrix.cuh"
#include "Timer.cuh"

#define SHARED_BYTES 49152
extern __shared__ float shared[];
typedef unsigned int uint;

struct MatrixSet {
    Matrix A, B, C;
};

struct Result {
    const char *version;
    Timer timer;
    explicit Result(const char *name): version(name) {}
};

__global__ void computeFromBlock(Matrix A, Matrix B, Matrix C) { // NOLINT(*-unnecessary-value-param)
    uint blockSize = blockDim.x;
    uint row = threadIdx.y;  // C-row that top-left of this sub_C maps to
    uint col = threadIdx.x;  // C-column that top-left of this sub_C maps to
    float *ptr_A = shared;
    float *ptr_B = &ptr_A[blockSize * blockSize];
    Matrix sub_C(C, blockSize, blockIdx.y, blockIdx.x);
    float sum = 0;

    for (int step = 0; step < A.cols / blockSize; ++step) {
        Matrix sub_A(A, blockSize, blockIdx.y, step);
        Matrix sub_B(B, blockSize, step, blockIdx.x);
        Matrix shared_A(ptr_A, blockSize);
        Matrix shared_B(ptr_B, blockSize);

        // Load sub-matrix A & B into shared memory
        shared_A.set(row, col, sub_A.get(row, col));
        shared_B.set(row, col, sub_B.get(row, col));

        // Use barrier to wait until the matrices are fully loaded
        __syncthreads();

        for (int i = 0; i < blockSize; ++i) {
            sum += __fmul_rn(shared_A.get(row, i), shared_B.get(i, col));
        }

        // Wait for aggregation to finish before loading the next step
        __syncthreads();
    }

    sub_C.set(row, col, sum);
}

Result computeFromDevice(const MatrixSet &m, int blockSize) {
    assert(m.A.cols == m.B.rows);

    Result result("Parallel CUDA");
    dim3 block(blockSize, blockSize);
    dim3 grid(m.C.cols / block.x, m.C.rows / block.y);
    size_t sharedMem = sizeof(float) * blockSize * blockSize;
    Matrix d_A(m.A.rows, m.A.cols, false);
    Matrix d_B(m.B.rows, m.B.cols, false);
    Matrix d_C(m.C.rows, m.C.cols, false);

    cudaMalloc(&d_A.ptr, d_A.size());
    cudaMalloc(&d_B.ptr, d_B.size());
    cudaMalloc(&d_C.ptr, d_C.size());
    cudaMemcpy(d_A.ptr, m.A.ptr, d_A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.ptr, m.B.ptr, d_B.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C.ptr, m.C.ptr, d_C.size(), cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(computeFromBlock, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES);

    result.timer.reset();
    computeFromBlock<<<grid, block, sharedMem * 2>>>(d_A, d_B, d_C);
    result.timer.stop();
    cudaDeviceSynchronize();

    cudaMemcpy(m.C.ptr, d_C.ptr, d_C.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_A.ptr);
    cudaFree(d_B.ptr);
    cudaFree(d_C.ptr);

    cudaCheckError();
    return result;
}

Result computeFromHost(const MatrixSet &m) {
    assert(m.A.cols == m.B.rows);

    Result result("Parallel CPU");

#pragma omp parallel
    for (int i = 0; i < m.C.rows * m.C.cols; ++i) {
        uint row = i / m.C.cols;
        uint col = i % m.C.cols;
        float sum = 0;

        for (int j = 0; j < m.A.cols; ++j) {
            sum += m.A.get(row, j) * m.B.get(j, col);
        }

        m.C.set(row, col, sum);
    }

    result.timer.stop();
    return result;
}

template<typename T, typename... U>
void measure(T func, U... args) {
    Result result = func(args...);
    double time = result.timer.elapsed();
    printf("%s: %.1f ms\n", result.version, time);
}

void compare(const Matrix &m1, const Matrix &m2) {
    const float epsilon = std::numeric_limits<float>::epsilon();
    int tolerance = 30;

    assert(m1.rows == m2.rows);
    assert(m1.cols == m2.cols);

    for (int r = 0; r < m1.rows; ++r) {
        for (int c = 0; c < m1.cols; ++c) {
            float x = m1.get(r, c);
            float y = m2.get(r, c);

            if (std::abs(x - y) > std::max(std::abs(x), std::abs(y)) * epsilon) {
                if (tolerance-- > 0) {
                    printf("Discrepancy at (%d, %d): %.7f, %.7f\n", r, c, x, y);
                } else {
                    printf("...\n");
                    return;
                }
            }
        }
    }
}

int main() {
    Matrix A(1024, 512);
    Matrix B(512, 1024);
    Matrix C1(1024, 1024);
    Matrix C2(1024, 1024);
    MatrixSet m1{A, B, C1};
    MatrixSet m2{A, B, C2};
    int blockSize = 16;

    printDeviceInfo();

    A.fillRandom(-1, 1);
    B.fillRandom(-1, 1);
    A.print("A: ");
    B.print("B: ");
    measure(computeFromHost, m1);
    C1.print("C: ");
    measure(computeFromDevice, m2, blockSize);
    C2.print("C: ");
    compare(C1, C2);
    return 0;
}
