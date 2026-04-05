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
Matrix MM_1D(Matrix A, Matrix B, int p) {

    if ( p > A.get_rows() ) p = A.get_rows();
    omp_set_num_threads(p);
    int m = A.get_rows();
    int n = A.get_columns();

    Matrix C(m, p);

    #pragma omp parallel shared(A, B, C)
    {
        int i, j , k;
        int thread_num = omp_get_thread_num();

        int number_of_rows_per_thread = A.get_rows() / p;

        int start = thread_num * number_of_rows_per_thread;
        int end = thread_num * number_of_rows_per_thread + number_of_rows_per_thread;

        if (thread_num == p - 1) {
            end = A.get_rows();
        }

        for (i = start; i < end ; i++) {
            for (j = 0; j <  B.get_columns(); j++) {
                int temp = 0;
                for (k = 0; k < n; k++) {
                    int a = A.get_value_at(i, k);
                    int b = B.get_value_at(k, j);
                    int c = a * b;
                    temp += c;
                }
                C.set_value_at(i, j, temp);
            }
        }
      }
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

    Matrix c = MM_1D(a, b, 2);
    std::vector<int> v2 = {22, 28, 49, 64};

    std::cout << a << "\n";
    std::cout << b << "\n";
    std::cout << c << "\n";
    assert(c.get_data() == v2);

    return 0;
}
