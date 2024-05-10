// Matrix.cuh

#ifndef CUDA_MATRIX_CUH
#define CUDA_MATRIX_CUH

#include <vector>
#include <string>

typedef unsigned int uint;

class Matrix {
public:
    float *ptr;
    const uint rows;
    const uint cols;
    const uint stride;

    Matrix(uint rows, uint cols, bool autoGC = true);

    __device__
    Matrix(Matrix &m, uint blockSize, uint blockRow, uint blockCol);

    __device__
    Matrix(float *ptr, uint blockSize);

    __host__ __device__
    Matrix(const Matrix &copy);

    __host__ __device__
    ~Matrix();

    __host__ __device__
    float get(uint r, uint c) const;

    __host__ __device__
    void set(uint r, uint c, float data) const;

    __host__ __device__
    void add(uint r, uint c, float data) const;

    __host__ __device__
    size_t size() const;

    __host__ __device__
    void fill(float num) const;

    void fillRandom(float min, float max) const;

    __host__ __device__
    void print(const char *prefix = "Matrix: ", int maxCount = 10) const;

private:
    bool autoGC;
    uint row;
    uint col;

    __host__ __device__
    size_t index(uint r, uint c) const;
};

#endif // CUDA_MATRIX_CUH
