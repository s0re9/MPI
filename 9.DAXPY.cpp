#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 16)  // Vector size 2^16 (65536)

void daxpy_serial(double a, double* x, double* y) {
    for (int i = 0; i < N; i++) {
        x[i] = a * x[i] + y[i];
    }
}

void print_vector(double* vec, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", vec[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    int rank, size;
    double a = 2.5;  // Scalar multiplier

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            printf("Vector size must be divisible by the number of processes!\n");
        }
        MPI_Finalize();
        return 1;
    }

    int elements_per_proc = N / size;
    double *x = NULL, *y = NULL;
    double *local_x = (double*)malloc(elements_per_proc * sizeof(double));
    double *local_y = (double*)malloc(elements_per_proc * sizeof(double));

    if (!local_x || !local_y) {
        printf("Memory allocation failed at Rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Rank 0 initializes full vectors
    if (rank == 0) {
        x = (double*)malloc(N * sizeof(double));
        y = (double*)malloc(N * sizeof(double));

        if (!x || !y) {
            printf("Memory allocation failed at Rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < N; i++) {
            x[i] = i * 1.0;
            y[i] = (N - i) * 1.0;
        }
    }

    // **Scatter the data**
    MPI_Scatter(x, elements_per_proc, MPI_DOUBLE, local_x, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, elements_per_proc, MPI_DOUBLE, local_y, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // **Parallel computation**
    double start_time = MPI_Wtime();
    for (int i = 0; i < elements_per_proc; i++) {
        local_x[i] = a * local_x[i] + local_y[i];
    }
    double end_time = MPI_Wtime();
    double parallel_time = end_time - start_time;

    // **Gather results**
    MPI_Gather(local_x, elements_per_proc, MPI_DOUBLE, x, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // **Serial computation for comparison**
    double serial_time = 0.0;
    if (rank == 0) {
        double start_serial = MPI_Wtime();
        daxpy_serial(a, x, y);
        double end_serial = MPI_Wtime();
        serial_time = end_serial - start_serial;

        printf("Parallel Execution Time: %f seconds\n", parallel_time);
        printf("Serial Execution Time: %f seconds\n", serial_time);
        printf("Speedup: %f\n", serial_time / parallel_time);

        free(x);
        free(y);
    }

    free(local_x);
    free(local_y);
    MPI_Finalize();
    return 0;
}
