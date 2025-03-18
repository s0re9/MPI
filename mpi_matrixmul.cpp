#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 70  // Matrix size

// Function to allocate memory for a 2D matrix
double** allocate_matrix(int rows, int cols) {
    double** matrix = new double* [rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new double[cols];
    }
    return matrix;
}

// Function to free memory of a 2D matrix
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Function to multiply part of the matrix
void matrix_multiply_part(double** A, double** B, double** C, int start, int end) {
    for (int i = start; i < end; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = SIZE / size;
    int extra_rows = SIZE % size;  // Handle case where SIZE is not divisible by size
    int start = rank * rows_per_proc + (rank < extra_rows ? rank : extra_rows);
    int end = start + rows_per_proc + (rank < extra_rows);

    // Allocate memory dynamically
    double** A = allocate_matrix(SIZE, SIZE);
    double** B = allocate_matrix(SIZE, SIZE);
    double** C = allocate_matrix(SIZE, SIZE);
    double** local_A = allocate_matrix(rows_per_proc + 1, SIZE);
    double** local_C = allocate_matrix(rows_per_proc + 1, SIZE);

    // Process 0 initializes matrices
    if (rank == 0) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
        }
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(&B[0][0], SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter matrix A using local buffer
    MPI_Scatter(&A[0][0], rows_per_proc * SIZE, MPI_DOUBLE, &local_A[0][0], rows_per_proc * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Measure execution time
    double start_time = omp_get_wtime();
    matrix_multiply_part(local_A, B, local_C, 0, rows_per_proc);
    double run_time = omp_get_wtime() - start_time;

    // Gather results
    MPI_Gather(&local_C[0][0], rows_per_proc * SIZE, MPI_DOUBLE, &C[0][0], rows_per_proc * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Process 0 prints execution time
    if (rank == 0) {
        printf("MPI Execution Time: %lf seconds\n", run_time);
    }

    // Free allocated memory
    free_matrix(A, SIZE);
    free_matrix(B, SIZE);
    free_matrix(C, SIZE);
    free_matrix(local_A, rows_per_proc + 1);
    free_matrix(local_C, rows_per_proc + 1);

    MPI_Finalize();
    return 0;
}
