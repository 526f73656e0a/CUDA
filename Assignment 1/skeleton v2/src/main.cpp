/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : Assignment 1
///
///  project    : GPU Programming
///
///  description: matrix-matrix multiplication
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, system
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <iomanip>


typedef std::chrono::milliseconds TimeT;

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Serial version for Matrix-Matrix multiplication AxB = C
//! @A pointer for the first entry of the square matrix A
//! @B pointer for the first entry of the square matrix B
//! @C pointer for the first entry of the square matrix C
//! @size specifies the number of rows and columns
////////////////////////////////////////////////////////////////////////////////////////////////////
void
MatrixMultiplicationSerial(const double* A, const double* B,
	double* C, const unsigned int size, double* Temp)
{	//Since B is constant create a temporary Matrix
	//Transpose Matrix B on the Temp Matrxi 
	for (unsigned int i = 0; i < size; ++i) {
		for (unsigned int j = 0; j < size; ++j) {
			Temp[i * size + j] = B[j * size + i];
		}
	}
	//Matrix Multiplication of A*B 
	for (unsigned int i = 0; i < size; ++i) {
		for (unsigned int j = 0; j < size; ++j) {
			for (unsigned int p = 0; p < size; ++p)
				C[i * size + j] += A[i * size + p] * Temp[j * size + p];
		}
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//! Parallel version for Matrix-Matrix multiplication AxB = C
//!
////////////////////////////////////////////////////////////////////////////////////////////////////
void multiply_threading(double* C, const int num_threads, const double* A, const double* B, const unsigned int size, unsigned int tid, const int op, const int rest) {
	for (int row = (tid - 1) * op; row < (tid * op) + rest; ++row) {
		for (unsigned int j = 0; j < size; j++) {
			for (unsigned int p = 0; p < size; ++p) {
				//std::cout << "Value of C[" << row << "][" << j << "]is =" << C[row * size + j] << std::endl;
				C[row * size + j] += A[row * size + p] * B[j * size + p]; // Reminder that this B is the transposed matrix of our OG B
			}
		}
	}
}
int
main(int /*argc*/, char** /*argv*/) {

	const unsigned int size = 2048;

	//set the correct number of threads at disposal in your machine
	static const int num_threads = std::thread::hardware_concurrency();
	// input arrays
	double* A = (double*)malloc(sizeof(double) * size * size);
	double* B = (double*)malloc(sizeof(double) * size * size);
	// output array
	double* C = (double*)malloc(sizeof(double) * size * size);
	//Inverse for B
	double* Temp = (double*)malloc(sizeof(double) * size * size);


	for (unsigned int i = 0; i < size; i++) {
		for (unsigned int j = 0; j < size; j++) {
			A[i * size + j] = 1.;
			B[i * size + j] = 3.;
			// Fill Array with empty elements
			C[i * size + j] = 0.;
		}
	}
#if 0

	auto start = std::chrono::steady_clock::now();

	MatrixMultiplicationSerial(A, B, C, size, Temp);

	double time = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start).count();


#else

	std::cout << "Number of supported threads = " << num_threads << std::endl;

	auto start = std::chrono::steady_clock::now();
	// init threads
	std::thread threads = new std::thread [num_threads];

	//how many rows each thread has to calc 
	const int op = size / num_threads;
	//rest of rows are taken by last thread 
	const int opd = size % num_threads;
	//Inverse Matrix B into Temp so that there is no time difference, between this and serial
	for (unsigned int i = 0; i < size; ++i) {
		for (unsigned int j = 0; j < size; ++j) {
			Temp[i * size + j] = B[j * size + i];
		}
	}
	//parallel version with multiple threads
	for (int i = 0; i < num_threads; ++i) {
		// so we dont have row = -1 in multiply_threading 
		if (i == num_threads - 1) {
			threads[i] = std::thread(multiply_threading, C, num_threads, A, Temp, size, i + 1, op, opd);
		}
		else
			threads[i] = std::thread(multiply_threading, C, num_threads, A, Temp, size, i + 1, op, 0);
	}
	//main thread waits for other threads to finish work 
	for (int i = 0; i < num_threads; ++i) {
		//std::cout << "Joining thread " << i << std::endl;
		threads[i].join();
	}
	//exec time recording
	double time = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start).count();
#endif
	//print exec time 
	std::cout << time << " ms" << std::endl;

	// correctness check
	bool correct = true;
	for (unsigned int k = 0; k < size * size; k++) {
		correct = (C[k] == 3. * size);
	}

	std::cout << " Matrix-Matrix multiplication is " << ((correct) ? "correct." : "incorrect.") << std::endl;
	// clean up memory
	free(A);
	free(B);
	free(C);
	free(Temp);
	return EXIT_SUCCESS;
}
