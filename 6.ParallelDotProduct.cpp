#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000  // Vector size

int main(int argc, char* argv[]) {
    int rank, size;
    double local_dot = 0.0, global_dot = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements_per_proc = N / size;  // Divide workload

    // **Allocate memory for local vectors**
    double* A = (double*)malloc(elements_per_proc * sizeof(double));
    double* B = (double*)malloc(elements_per_proc * sizeof(double));

    // **Master process initializes full vectors**
    if (rank == 0) {
        double* full_A = (double*)malloc(N * sizeof(double));
        double* full_B = (double*)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            full_A[i] = 1.0;  // Example: All elements are 1.0
            full_B[i] = 1.0;
        }
        // **Scatter data to all processes**
        MPI_Scatter(full_A, elements_per_proc, MPI_DOUBLE, A, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(full_B, elements_per_proc, MPI_DOUBLE, B, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(full_A);
        free(full_B);
    } else {
        MPI_Scatter(NULL, elements_per_proc, MPI_DOUBLE, A, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, elements_per_proc, MPI_DOUBLE, B, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // **Each process computes partial dot product**
    for (int i = 0; i < elements_per_proc; i++) {
        local_dot += A[i] * B[i];
    }

    // **Reduce: Sum all local dot products**
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // **Master prints final result**
    if (rank == 0) {
        printf("Dot Product: %f\n", global_dot);
    }

    free(A);
    free(B);
    MPI_Finalize();
    return 0;
}
