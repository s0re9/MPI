#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000  // Size of the array

int main(int argc, char* argv[]) {
    int rank, size;
    double local_sum = 0.0, global_sum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements_per_proc = N / size;  // Divide work among processes
    double* data = NULL;

    if (rank == 0) {
        // **Master Process: Initialize Data**
        data = (double*)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            data[i] = 1.0;  // Example: all values are 1.0
        }
    }

    // **Allocate Memory for Each Process' Chunk**
    double* local_data = (double*)malloc(elements_per_proc * sizeof(double));

    // **Scatter the array among processes**
    MPI_Scatter(data, elements_per_proc, MPI_DOUBLE,
        local_data, elements_per_proc, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // **Each Process Computes Local Sum**
    for (int i = 0; i < elements_per_proc; i++) {
        local_sum += local_data[i];
    }

    // **Reduce: Compute Global Sum at Rank 0**
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // **Master Process Prints the Result**
    if (rank == 0) {
        printf("Total Sum: %f\n", global_sum);
        free(data);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}
