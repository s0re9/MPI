#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 70  // Matrix size

void matrix_multiply_serial(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];

    // Initialize matrices A and B with random values
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    // Measure execution time
    double start_time = omp_get_wtime();
    matrix_multiply_serial(A, B, C);
    double run_time = omp_get_wtime() - start_time;

    printf("Serial Execution Time: %lf seconds\n", run_time);
    return 0;
}
