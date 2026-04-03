// Comparison of the performances of serial and parallel
// Matrix Multiplication algorithm implementations

#include <iostream>
#include <vector>
#include <omp.h>
#include "matrix.hpp"
#include <algorithm>

// Serial algorithm
// Matrix A (dims mxn) * Matrix B (dims nxp) = Matrix C (dims mxp)
Matrix MM_ser(Matrix A, Matrix B) {
    Matrix c(A.get_rows(), B.get_columns());

    return c;
}

// Simple parallel algorithm
void MM_Par() {}

// 1D Parallel algorithm
void MM_1D() {}

// 2D Parallel algorithm
void MM_2D() {}

int main()
{

    // std::vector<int> v = { 1,2,3,4,5,6 };
    // Matrix a(2, 3, v);
    // Matrix b(3, 2, v);
    // Matrix c;

    // c = MM_ser(a, b);
    // std::vector<int> v2 = { 22, 28, 49,  64 };

    // assert(c.get_dat() == v2);

    return 0;
}