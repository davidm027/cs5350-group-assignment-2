// Comparison of the performances of serial and parallel
// Matrix Multiplication algorithm implementations

#include <cassert>
#include <iostream>
#include <vector>

#include <omp.h>

#include "matrix.hpp"

// Serial algorithm
// Matrix A (dims mxn) * Matrix B (dims nxp) = Matrix C (dims mxp)
Matrix MM_ser(Matrix A, Matrix B) {
    int m = A.get_rows();
    int n = A.get_columns();
    int p = B.get_columns();

    Matrix C(m, p);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            int temp = 0;
            for (int k = 0; k < n; k++) {
                int a = A.get_value_at(i, k);
                int b = B.get_value_at(k, j);
                int c = a * b;
                temp += c;
            }
            C.set_value_at(i, j, temp);
        }
    }
    return C;
}

// Simple parallel algorithm
Matrix MM_Par(Matrix A, Matrix B) {
    int m = A.get_rows();
    int n = A.get_columns();
    int p = B.get_columns();

    Matrix C(m, p);

#pragma omp parallel for collapse(3)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            int temp = 0;
            for (int k = 0; k < n; k++) {
                int a = A.get_value_at(i, k);
                int b = B.get_value_at(k, j);
                int c = a * b;
                temp += c;
            }
            C.set_value_at(i, j, temp);
        }
    }
    return C;
}

// 1D Parallel algorithm
Matrix MM_1D(Matrix A, Matrix B) {
    int m = A.get_rows();
    int n = A.get_columns();
    int p = B.get_columns();

    Matrix C(m, p);

    return C;
}

// 2D Parallel algorithm
Matrix MM_2D(Matrix A, Matrix B) {
    int m = A.get_rows();
    int n = A.get_columns();
    int p = B.get_columns();

    Matrix C(m, p);

    return C;
}

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6};
    Matrix a(2, 3, v);
    Matrix b(3, 2, v);

    Matrix c = MM_ser(a, b);
    std::vector<int> v2 = {22, 28, 49, 64};

    std::cout << a << "\n";
    std::cout << b << "\n";
    std::cout << c << "\n";
    assert(c.get_data() == v2);

    return 0;
}