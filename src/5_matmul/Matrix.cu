// Matrix.cu

#include <string>
#include <random>
#include "Matrix.cuh"

Matrix::Matrix(uint rows, uint cols, bool autoGC) :
        rows(rows), cols(cols),
        row(0), col(0),
        stride(cols),
        autoGC(autoGC) {
    ptr = new float[sizeof(float) * rows * cols];
    fill(0);
}

__device__
Matrix::Matrix(Matrix &m, uint blockSize, uint blockRow, uint blockCol) :
        ptr(m.ptr),
        rows(blockSize), cols(blockSize),
        row(blockRow * blockSize), col(blockCol * blockSize),
        stride(m.stride),
        autoGC(false) {}

__device__
Matrix::Matrix(float *ptr, uint blockSize) :
        ptr(ptr),
        rows(blockSize), cols(blockSize),
        row(0), col(0),
        stride(blockSize),
        autoGC(false) {}

__host__ __device__
Matrix::Matrix(const Matrix &copy) :
        ptr(copy.ptr),
        rows(copy.rows), cols(copy.cols),
        row(copy.row), col(copy.col),
        stride(copy.stride),
        autoGC(false) {}

__host__ __device__
Matrix::~Matrix() {
    if (autoGC) {
        free(ptr);
    }
}

__host__ __device__
void Matrix::fill(float num) const {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            set(r, c, num);
        }
    }
}

void Matrix::fillRandom(float min, float max) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            set(r, c, dist(gen));
        }
    }
}

__host__ __device__
float Matrix::get(uint r, uint c) const {
    return ptr[index(r, c)];
}

__host__ __device__
void Matrix::set(uint r, uint c, float data) const {
    ptr[index(r, c)] = data;
}

__host__ __device__
void Matrix::add(uint r, uint c, float data) const {
    ptr[index(r, c)] += data;
}

__host__ __device__
size_t Matrix::size() const {
    return sizeof(float) * stride * rows;
}

__host__ __device__
void Matrix::print(const char *prefix, int maxCount) const {
    int len = static_cast<int>(rows * cols);
    int remainder = len - maxCount;
    int count = maxCount < len ? maxCount : len;

    printf("%s", prefix);

    for (int i = 0; i < count; ++i) {
        uint r = i / cols;
        uint c = i % cols;
        printf("%.4f ", get(r, c));
    }

    if (remainder > 0) {
        printf("... %d more items\n", remainder);
    } else {
        printf("\n");
    }
}

__host__ __device__
size_t Matrix::index(uint r, uint c) const {
    return (r + row) * stride + c + col;
}
