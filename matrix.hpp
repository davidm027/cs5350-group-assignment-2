#include <vector>
#include <string>

class Matrix {
private:
    int rows;
    int columns;
    std::vector<int> data;
public:
    Matrix(int, int);
    Matrix(int, int, std::vector<int>);
    ~Matrix();
    int get_rows();
    int get_columns();
    std::vector<int> get_data();
    int get_value_at(int, int);
    void set_value_at(int, int, int);
    friend std::ostream& operator<<(std::ostream&, Matrix);
};