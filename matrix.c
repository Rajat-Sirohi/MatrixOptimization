#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
 */

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows < 1 || cols < 1) {
	PyErr_SetString(PyExc_TypeError, "Dimensions must be positive!");
	return -1;
    }
    
    *mat = (matrix*) malloc(sizeof(matrix));
    if (!*mat) {
	PyErr_SetString(PyExc_RuntimeError, "Allocation error!");
	return -1;
    }
    
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->data = (double*) calloc(rows * cols, sizeof(double));
    if (!(*mat)->data) {
	PyErr_SetString(PyExc_RuntimeError, "Allocation error!");
	return -1;
    }
    
    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;
    
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    if (rows < 1 || cols < 1) {
	PyErr_SetString(PyExc_TypeError, "Dimensions must be positive!");
	return -1;
    }
    if ((from->rows)*(from->cols) - offset < rows * cols) {
	PyErr_SetString(PyExc_TypeError, "Offset too large, overflow from!");
	return -1;
    }
    
    *mat = (matrix*) malloc(sizeof(matrix));
    if (!*mat) {
	PyErr_SetString(PyExc_RuntimeError, "Allocation error!");
	return -1;
    }
    
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->data = from->data + offset;
    
    from->ref_cnt += 1;
    (*mat)->ref_cnt = 1;
    (*mat)->parent = from;
    
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    if (mat) {
	mat->ref_cnt -= 1;
    	if (mat->parent)
    	    mat->parent->ref_cnt -= 1;
	
    	if (!mat->parent && mat->ref_cnt == 0) {
    	    free(mat->data);
	    free(mat);
	}
	else if (mat->parent && mat->parent->ref_cnt == 0) {
    	    free(mat->parent->data);
	    free(mat->parent);
	    free(mat);
    	}
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return mat->data[col + mat->cols * row];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[col + mat->cols * row] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    for (int i = 0; i < (mat->rows)*(mat->cols); i++) {
	mat->data[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int rows1 = mat1->rows, cols1 = mat1->cols;
    int rows2 = mat2->rows, cols2 = mat2->cols;
    int rowsr = result->rows, colsr = result->cols;
    if (rows1 != rows2 || rows1 != rowsr ||
	cols1 != cols2 || cols1 != colsr)
	return -1;
    
    int n = rows1 * cols1;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
	result->data[i] = mat1->data[i] + mat2->data[i];
    }
    
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int rows1 = mat1->rows, cols1 = mat1->cols;
    int rows2 = mat2->rows, cols2 = mat2->cols;
    int rowsr = result->rows, colsr = result->cols;
    if (rows1 != rows2 || rows1 != rowsr ||
	cols1 != cols2 || cols1 != colsr)
	return -1;
    
    int n = rows1 * cols1;
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
	result->data[i] = mat1->data[i] - mat2->data[i];
    
    return 0;
}

/*
 * Matrix transpose with cache blocking; helper function for mul_matrix
 */
void transpose(double *orig, double *trans, int rows, int cols) {
    const int BLOCKSIZE = 20;
    const int SIZE = rows * cols;
    #pragma omp parallel
    {
	int x,y,xx,yy,i,j;
	#pragma omp for
	for (x = 0; x < rows; x += BLOCKSIZE) {
	    for (y = 0; y < cols; y += BLOCKSIZE) {
		for (xx = 0; xx < BLOCKSIZE; xx++) {
		    for (yy = 0; yy < BLOCKSIZE; yy++) {
			i = x + xx;
			j = y + yy;
			if (i + j*rows < SIZE &&
			    j + i*cols < SIZE)
			    trans[i + j*rows] = orig[j + i*cols];
		    }
		}
	    }
	}
    }
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int rows1 = mat1->rows, cols1 = mat1->cols;
    int rows2 = mat2->rows, cols2 = mat2->cols;
    int rowsr = result->rows, colsr = result->cols;
    if (cols1 != rows2 || rowsr != rows1 || colsr != cols2)
	return -1;
    
    double *trans = (double*)malloc(sizeof(double) * mat2->rows * mat2->cols);
    transpose(mat2->data, trans, mat2->rows, mat2->cols);
    #pragma omp parallel
    {
        int i, j, k;
	double *sumArr = malloc(4 * sizeof(double));
        #pragma omp for
        for (i = 0; i < rows1; i++) {
            for (j = 0; j < cols2; j++) {
		__m256d dotp[4] = {_mm256_setzero_pd()};
                for (k = 0; k < cols1 - 16; k += 16) {
		    __m256d product = _mm256_mul_pd(_mm256_loadu_pd(mat1->data + k + i*cols1),
						    _mm256_loadu_pd(trans + k + j*rows2));
		    dotp[0] = _mm256_add_pd(dotp[0], product);
		    
		    product = _mm256_mul_pd(_mm256_loadu_pd(mat1->data + k + 4 + i*cols1),
					    _mm256_loadu_pd(trans + k + 4 + j*rows2));
		    dotp[1] = _mm256_add_pd(dotp[1], product);
		    
		    product = _mm256_mul_pd(_mm256_loadu_pd(mat1->data + k + 8 + i*cols1),
					    _mm256_loadu_pd(trans + k + 8 + j*rows2));
		    dotp[2] = _mm256_add_pd(dotp[2], product);
		    
		    product = _mm256_mul_pd(_mm256_loadu_pd(mat1->data + k + 12 + i*cols1),
					    _mm256_loadu_pd(trans + k + 12 + j*rows2));
		    dotp[3] = _mm256_add_pd(dotp[3], product);
                }
		_mm256_storeu_pd(sumArr, dotp[0]);
		double final = sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];
		_mm256_storeu_pd(sumArr, dotp[1]);
		final += sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];
		_mm256_storeu_pd(sumArr, dotp[2]);
		final += sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];
		_mm256_storeu_pd(sumArr, dotp[3]);
		final += sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];
		
		for (; k < cols1; k++) {
		    final += mat1->data[k + i*cols1] * trans[k + j*rows2];
		}
		result->data[j + i*colsr] = final;
            }
        }
	free(sumArr);
    }
    free(trans);
    
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 *
 * Implements Exponentiation by Squaring method
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    if (mat->rows != mat->cols)
	return -1;
    if (pow < 0)
	return -2;
    const int N = mat->rows;
    
    matrix *temp1 = NULL, *temp2 = NULL;
    allocate_matrix(&temp1, N, N);
    allocate_matrix(&temp2, N, N);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
	for (int j = 0; j < N; j++) {
	    temp1->data[j + i*N] = mat->data[j + i*N];
	    temp2->data[j + i*N] = (i == j);
	    result->data[j + i*N] = (i == j);
	}
    }
    
    if (pow == 0) {
	deallocate_matrix(temp1);
	deallocate_matrix(temp2);
	return 0;
    }
    
    matrix *square = NULL, *product = NULL;
    allocate_matrix(&square, N, N);
    allocate_matrix(&product, N, N);
    int powOdd = -1;
    while (pow > 1) {
	mul_matrix(square, temp1, temp1);
	if (pow % 2 == 0) {
	    powOdd = 0;
	    pow /= 2;
	} else {
	    mul_matrix(product, temp1, temp2);
	    powOdd = 1;
	    pow = (pow - 1) / 2;
	}
	
	#pragma omp parallel for
	for (int i = 0; i < N*N; i++) {
	    temp1->data[i] = square->data[i];
	    if (powOdd)
		temp2->data[i] = product->data[i];
	}
    }
    
    mul_matrix(result, temp1, temp2);
    deallocate_matrix(square);
    deallocate_matrix(product);
    deallocate_matrix(temp1);
    deallocate_matrix(temp2);
    
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    int rows = mat->rows, cols = mat->cols;
    int rowsr = result->rows, colsr = result->cols;
    if (rows != rowsr || cols != colsr)
	return -1;
    
    int n = rows * cols;
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
	result->data[i] = -(mat->data[i]);
    
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    int rows = mat->rows, cols = mat->cols;
    int rowsr = result->rows, colsr = result->cols;
    if (rows != rowsr || cols != colsr)
	return -1;
    
    int n = rows * cols;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
	double x = mat->data[i];
	result->data[i] = (x < 0) ? -x : x;
    }
    
    return 0;
}