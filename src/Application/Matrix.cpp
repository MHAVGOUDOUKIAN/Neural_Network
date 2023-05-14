#include "Matrix.h"

Matrix::Matrix(std::size_t row, std::size_t col)
: m_row(row)
, m_col(col)
, m_mat(m_row, std::vector<float>(m_col, 0.f))
{
	for(int i=0; i<m_row; i++) {
		for(int j=0; j<m_col; j++) {
			m_mat[i][j] = randomf(0,1);
		}	
	}
}

Matrix::Matrix(std::size_t row, std::size_t col, float value)
: m_row(row)
, m_col(col)
, m_mat(m_row, std::vector<float>(m_col, value))
{
}


Matrix::Matrix(std::vector<float> values, bool isDiag = false)
{
	if(isDiag)
	{
		m_row = values.size();
		m_col = values.size();
		m_mat = std::vector<std::vector<float>>(m_row, std::vector<float>(m_col, 0.f));
		
		for(std::size_t i = 0; i < m_row; ++i)
		{
			m_mat[i][i] = values[i];
		}
	}
	else
	{
		m_row = values.size();
		m_col = 1;
		m_mat = std::vector<std::vector<float>>(m_row, std::vector<float>(m_col, 0.f));
		
		for(std::size_t i = 0; i < m_row; ++i)
		{
			m_mat[i][0] = values[i];
		}
	}
}

void Matrix::setCoeff(std::size_t i, std::size_t j, const float newValue)
{
	m_mat[i][j] = newValue;
}

float Matrix::getCoeff(std::size_t i, std::size_t j) const
{
	return(m_mat[i][j]);
}

std::size_t Matrix::row() const
{
	return(m_row);
}

std::size_t Matrix::col() const
{
	return(m_col);
}

Matrix Matrix::operator+(const Matrix& B) const
{
	//Vérification de la possibilité d'addition
	std::size_t rowA = this->row();
	std::size_t colA = this->col();
	std::size_t rowB = B.row();
	std::size_t colB = B.col();
	
	if (rowA != rowB || colA != colB)
		throw std::runtime_error ("Matrix::operator+ - wrong dimensions");
	
	std::size_t row = rowA;
	std::size_t col = colA;
	
	//Addition
	Matrix mat(row, col);
	
	for(std::size_t i = 0; i < row; ++i)
	{
		for(std::size_t j = 0; j < col; ++j)
		{
			mat.setCoeff(i, j, B.getCoeff(i, j) + this->getCoeff(i, j));
		}
	}
	
	return(mat);
}

Matrix Matrix::operator*(const Matrix& B) const
{
	//Vérification de la possibilité de multiplication
	std::size_t rowA = this->row();
	std::size_t colA = this->col();
	std::size_t rowB = B.row();
	std::size_t colB = B.col();
	
	if (colA != rowB)
		throw std::runtime_error ("Matrix::operator* - wrong dimensions");
	
	std::size_t temp = colA;
	std::size_t row = rowA;
	std::size_t col = colB;
	
	//Multiplication
	
	Matrix mat(row, col);
	
	for(std::size_t i = 0; i < row; ++i)
	{
		for(std::size_t j = 0; j < col; ++j)
		{
			float coef = 0.f;
			for(std::size_t k = 0; k < temp; ++k)
			{
				coef += this->getCoeff(i, k) * B.getCoeff(k, j);
			}
			
			mat.setCoeff(i, j, coef);
		}
	}
	
	return(mat);
}

Matrix Hadamard(const Matrix& A, const Matrix& B) {
	std::size_t rowA = A.row();
	std::size_t colA = A.col();
	std::size_t rowB = B.row();
	std::size_t colB = B.col();
	
	if (rowA != rowB || colA != colB)
		throw std::runtime_error ("Matrix::Hadamard - wrong dimensions");

	Matrix res = Matrix(rowA, colA);

	for(std::size_t i = 0; i < rowA; ++i)
	{
		for(std::size_t j = 0; j < colA; ++j)
		{
			res.setCoeff(i, j, A.getCoeff(i, j) * B.getCoeff(i, j));
		}
	}

	return res;
}	

void Matrix::operator-=(const Matrix& B){
	std::size_t rowA = m_row;
	std::size_t colA = m_col;
	std::size_t rowB = B.row();
	std::size_t colB = B.col();
	
	if (rowA != rowB || colA != colB)
	{
		std::cout << rowA << " " << rowB << " " << colA << " " << colB << std::endl;
		throw std::runtime_error ("Matrix::operator-= - wrong dimensions");
	}
	
	std::size_t row = rowA;
	std::size_t col = colA;
	
	for(std::size_t i = 0; i < row; ++i)
	{
		for(std::size_t j = 0; j < col; ++j)
		{
			m_mat[i][j] -= B.getCoeff(i, j);
		}
	}
}

void Matrix::constMult(float c)
{
	for(std::size_t i = 0; i < m_row; ++i)
	{
		for(std::size_t j = 0; j < m_col; ++j)
		{
			m_mat[i][j] = c * m_mat[i][j];
		}
	}
}

void Matrix::applySigmo()
{
	for(std::size_t i = 0; i < m_row; ++i)
	{
		for(std::size_t j = 0; j < m_col; ++j)
		{
			m_mat[i][j] = 1.f / (1.f + exp(-1.f * m_mat[i][j]));
		}
	}
}

void Matrix::applySigmoPrime()
{
	for(std::size_t i = 0; i < m_row; ++i)
	{
		for(std::size_t j = 0; j < m_col; ++j)
		{
			float sigmo = 1.f / (1.f + exp(-1.f * m_mat[i][j]));
			m_mat[i][j] = sigmo * (1 - sigmo);
		}
	}
}

Matrix Matrix::transposee() const
{
	Matrix newMat(m_col, m_row);
	
	for (std::size_t i = 0; i < m_row; ++i)
	{
		for (std::size_t j = 0; j < m_col; ++j)
		{
			newMat.setCoeff(j, i, m_mat[i][j]);
		}
	}
	
	return(newMat);
}

Matrix Matrix::diag() const
{
	if (m_col != 1)
		throw std::runtime_error ("Matrix::diag - wrong dimensions");
	
	std::vector<float> vect;
	for(std::size_t i = 0; i < m_row; ++i)
	{
		vect.push_back(getCoeff(i, 0));
	}
	Matrix res(vect, true);
	
	return(res);
}

void Matrix::disp() const //Debug function
{
	for (std::size_t i = 0; i < m_row; ++i)
	{
		for (std::size_t j = 0; j < m_col; ++j)
		{
			std::cout << m_mat[i][j] << " / ";
		}
		std::cout << std::endl;
	}
	
	std::cout << "----------------------------------" <<  std::endl;
}

Matrix BroadCastAdd(const Matrix& A, const Matrix& B) {
	Matrix res = A;
	if(B.row() == 1 and B.col()==1) {
		for(std::size_t i = 0; i < A.row(); ++i)
		{
			for(std::size_t j = 0; j < A.col(); ++j)
			{
				res.setCoeff(i,j,A.getCoeff(i,j)+B.getCoeff(0,0));
			}
		}
	} else if (B.row()==1 and B.col()== A.col()) {
		for(std::size_t i = 0; i < A.row(); ++i)
		{
			for(std::size_t j = 0; j < A.col(); ++j)
			{
				res.setCoeff(i,j,A.getCoeff(i,j)+B.getCoeff(0,j));
			}
		}
	} else if (B.row()== A.row() and B.col()== 1) {
		for(std::size_t i = 0; i < A.row(); ++i)
		{
			for(std::size_t j = 0; j < A.col(); ++j)
			{
				res.setCoeff(i,j,A.getCoeff(i,j)+B.getCoeff(i,0));
			}
		}
	} else throw std::runtime_error ("Matrix::BroadCastAdd - wrong dimensions");
	return res;
}

Matrix SumOnCol(const Matrix& A) {
	Matrix res = Matrix(A.row(),1, 0.0f);
	for(std::size_t i = 0; i < A.row(); ++i)
	{
		for(std::size_t j = 0; j < A.col(); ++j)
		{
			res.setCoeff(i,0,A.getCoeff(i,j)+res.getCoeff(i,0));
		}
	}
	return res;
}