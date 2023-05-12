#ifndef MATRIX_H
#define MATRIX_H

#include <math.h>
#include <vector>
#include <iostream>

class Matrix
{
public :
	
	Matrix(std::size_t, std::size_t);
	Matrix(std::size_t, std::size_t, float value);
	Matrix(const std::vector<float>, bool);
	
	void setCoeff(std::size_t, std::size_t, const float);
	float getCoeff(std::size_t, std::size_t) const;
	std::size_t row() const;
	std::size_t col() const;
	
	Matrix operator+(const Matrix&) const;
	Matrix operator*(const Matrix&) const;
	void operator-=(const Matrix&);
	
	Matrix transposee() const;
	Matrix diag() const;
	
	void constMult(float);
	void applySigmo();
	void applySigmoPrime();
	
	void disp() const; // Debug function
	
private :
	
	std::size_t m_row;
	std::size_t m_col;
	std::vector<std::vector<float>> m_mat;
};

Matrix Hadamard(const Matrix& A, const Matrix& B);
Matrix BroadCastAdd(const Matrix& A, const Matrix& B);
Matrix SumOnCol(const Matrix& A);

#endif //MATRIX_H
