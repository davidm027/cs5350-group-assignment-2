// Comparison of the performances of serial and parallel
// Matrix Multiplication algorithm implementations

#include <cassert>
#include <iostream>
#include <vector>

#include <omp.h>

#include "matrix.hpp"
#include <cmath>


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
    int b_columns = B.get_columns();
    int number_of_rows_per_thread = A.get_rows() / p;


    Matrix C(m, b_columns);

    #pragma omp parallel shared(A, B, C)
    {
        int i, j , k;
        int thread_num = omp_get_thread_num();

        int start = thread_num * number_of_rows_per_thread;
        int end = thread_num * number_of_rows_per_thread + number_of_rows_per_thread;

        if (thread_num == p - 1) {
            end = A.get_rows();
        }

        for (i = start; i < end ; i++) {
            for (j = 0; j <  b_columns; j++) {
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
Matrix MM_2D(Matrix A, Matrix B, int p) {
    int m1 = A.get_rows();
    int n1 = A.get_columns();
    int m2 = B.get_rows();
    int n2 = B.get_columns();

    Matrix C(m1, n2);
    int thread_dim = (int) std::sqrt(p);
    int number_of_rows_per_thread = A.get_rows() / thread_dim;
    int number_of_columns_per_thread = B.get_columns() / thread_dim;

    // if ( p_rows > m1 ) p_rows = m1;
    // if ( p_columns >  n2 ) p_columns = n2;

    omp_set_num_threads(p);

    int sum = 0;

    #pragma omp parallel shared(A, B, C)
    {
        int i, j , k;
        int thread_num = omp_get_thread_num();

        int row = thread_num / thread_dim;
        int col = thread_num % thread_dim;

        int start = row * number_of_rows_per_thread;
        int end = row * number_of_rows_per_thread + number_of_rows_per_thread;

        int column_start = col * number_of_columns_per_thread;



        if (row == thread_dim - 1) {
            end = A.get_rows();
        }

        int end_column = column_start + number_of_columns_per_thread;

        if (col == thread_dim -1) {
            end_column = B.get_columns();
        }


        for (i = start; i < end; i++) {
            for (j = column_start; j <  end_column; j++) {
                int temp = 0;

                int k_start = col * (n1 / thread_dim);
                int k_end = k_start + (n1 / thread_dim);
                for (k = k_start; k < k_end; k++) {
                    int a = A.get_value_at(i, k);
                    int b = B.get_value_at(k, j);
                    int c = a * b;
                    temp += c;
                }
                #pragma omp critical
                {
                temp += C.get_value_at(i, j);
                C.set_value_at(i, j, temp);
                }


            }
        }
    }
    return C;
}

int main() {
    std::vector<int> v = {1, 2, 4, 5};
    Matrix a(2, 2, v);
    Matrix b(2, 2, v);

    Matrix c = MM_2D(a, b, 4);
    std::vector<int> v2 = {1, 10, 4, 25};

    std::cout << a << "\n";
    std::cout << b << "\n";
    std::cout << c << "\n";
    assert(c.get_data() == v2);

    return 0;
}
