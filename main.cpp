// Comparison of the performances of serial and parallel
// Matrix Multiplication algorithm implementations

#include <omp.h>

#include <boost/program_options.hpp>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "ctrack.hpp"
#include "matrix.hpp"

// Serial algorithm
// Matrix A (dims mxn) * Matrix B (dims nxp) = Matrix C (dims mxp)
Matrix MM_ser(Matrix A, Matrix B) {
    CTRACK;
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
    CTRACK;
    int m = A.get_rows();
    int n = A.get_columns();
    int p = B.get_columns();

    Matrix C(m, p);

    #pragma omp parallel for
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
    CTRACK;
    if (p > A.get_rows())
        p = A.get_rows();
    omp_set_num_threads(p);
    int m = A.get_rows();
    int n = A.get_columns();
    int b_columns = B.get_columns();
    int number_of_rows_per_thread = A.get_rows() / p;

    Matrix C(m, b_columns);

    #pragma omp parallel shared(A, B, C)
    {
        int i, j, k;
        int thread_num = omp_get_thread_num();

        int start = thread_num * number_of_rows_per_thread;
        int end =
            thread_num * number_of_rows_per_thread + number_of_rows_per_thread;

        if (thread_num == p - 1) {
            end = A.get_rows();
        }

        for (i = start; i < end; i++) {
            for (j = 0; j < b_columns; j++) {
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
    CTRACK;
    int m1 = A.get_rows();
    int n1 = A.get_columns();
    int m2 = B.get_rows();
    int n2 = B.get_columns();

    Matrix C(m1, n2);
    int thread_dim = (int)std::sqrt(p);
    int number_of_rows_per_thread = A.get_rows() / thread_dim;
    int number_of_columns_per_thread = B.get_columns() / thread_dim;

    // if ( p_rows > m1 ) p_rows = m1;
    // if ( p_columns >  n2 ) p_columns = n2;

    omp_set_num_threads(p);

    int sum = 0;

    #pragma omp parallel shared(A, B, C)
    {
        int i, j, k;
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

        if (col == thread_dim - 1) {
            end_column = B.get_columns();
        }

        for (i = start; i < end; i++) {
            for (j = column_start; j < end_column; j++) {
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

Matrix MM_2D_second_version(Matrix A, Matrix B, int p) {
    CTRACK;
    int m1 = A.get_rows();
    int n1 = A.get_columns();
    int m2 = B.get_rows();
    int n2 = B.get_columns();
    Matrix C(m1, n2);
    int thread_dim = (int)std::sqrt(p);
    int number_of_rows_per_thread = A.get_rows() / thread_dim;
    int number_of_columns_per_thread = B.get_columns() / thread_dim;
    omp_set_num_threads(p);

    #pragma omp parallel shared(A, B, C)
    {
        int i, j, k;
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
        if (col == thread_dim - 1) {
            end_column = B.get_columns();
        }

        int local_rows = end - start;
        int local_cols = end_column - column_start;

        std::vector<std::vector<int>> local(local_rows,
                                            std::vector<int>(local_cols));

        for (i = start; i < end; i++) {
            for (j = column_start; j < end_column; j++) {
                int temp = 0;
                for (k = 0; k < n1; k++) {
                    temp += A.get_value_at(i, k) * B.get_value_at(k, j);
                }
                local[(i - start)][(j - column_start)] = temp;
            }
        }

        for (i = start; i < end; i++) {
            for (j = column_start; j < end_column; j++) {
                C.set_value_at(i, j, local[(i - start)][(j - column_start)]);
            }
        }
    }
    return C;
}

Matrix create_random_matrix(int rows, int columns, unsigned int seed = 5350) {
    std::mt19937 rng(seed);
    std::vector<int> v;

    int size = rows * columns;
    for (int i = 0; i < size; i++) {
        int r = (int)rng() % 1000;
        v.push_back(r);
    }

    Matrix C(rows, columns, v);

    return C;
}

int main(int argc, const char* argv[]) {
    // set up CLI args (makes it easier to run as a script)
    int m, n, q, P;

    std::string fname;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")(
        "rows-A,m", po::value<int>(), "set amount of rows for matrix A")(
        "columns-A,n", po::value<int>(), "set amount of columns for matrix A")(
        "columns-B,q", po::value<int>(), "set amount of columns for matrix B")(
        "processors,P", po::value<int>(),
        "set number of processors for the parallel algorithms to use")(
        "output-file,o", po::value<std::string>()->default_value("results.txt"),
        "name of file containing ctrack output");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << '\n';
    } else {
        if (vm.count("rows-A") && vm.count("columns-A") &&
            vm.count("columns-B") && vm.count("processors")) {
            m = vm["rows-A"].as<int>();
            n = vm["columns-A"].as<int>();
            q = vm["columns-B"].as<int>();
            P = vm["processors"].as<int>();

            if (vm.count("rows-A")) {
                fname = vm["output-file"].as<std::string>();
            }
        } else {
            std::cout << "Not all variables were set.\n";
            return -1;
        }
    }

    // actual code to run everything
    Matrix a = create_random_matrix(m, n);
    Matrix b = create_random_matrix(n, q);
    Matrix c1 = MM_ser(a, b);
    Matrix c2 = MM_Par(a, b);
    Matrix c3 = MM_1D(a, b, P);
    Matrix c4 = MM_2D_second_version(a, b, P);

    std::string results = ctrack::result_as_string();
    std::ofstream out;
    out.open(fname);
    out << "m: " << m << ", ";
    out << "n: " << n << ", ";
    out << "q: " << q << ", ";
    out << "P: " << P << "\n";
    out << results;
    out.close();

    return 0;
}