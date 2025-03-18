#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4  // Matrix size (NxN)

void print_matrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            printf("Matrix size must be divisible by the number of processes!\n");
        }
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / size;

    // **Dynamic memory allocation**
    int* matrix = NULL;
    int* transposed = NULL;
    int* local_rows = (int*)malloc(rows_per_proc * N * sizeof(int));
    int* local_transposed = (int*)malloc(rows_per_proc * N * sizeof(int));

    if (!local_rows || !local_transposed) {
        printf("Memory allocation failed at Rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        matrix = (int*)malloc(N * N * sizeof(int));
        transposed = (int*)malloc(N * N * sizeof(int));

        if (!matrix || !transposed) {
            printf("Memory allocation failed at Rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // **Initialize the matrix**
        int count = 1;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = count++;
            }
        }
        printf("Original Matrix:\n");
        print_matrix(matrix, N, N);
    }

    // **Scatter rows of matrix to all processes**
    MPI_Scatter(matrix, rows_per_proc * N, MPI_INT, local_rows, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

    // **Each process transposes its local rows**
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            local_transposed[j * rows_per_proc + i] = local_rows[i * N + j];
        }
    }

    // **Gather results at Rank 0**
    MPI_Gather(local_transposed, rows_per_proc * N, MPI_INT, transposed, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

    // **Print the transposed matrix**
    if (rank == 0) {
        printf("Transposed Matrix:\n");
        print_matrix(transposed, N, N);
        free(matrix);
        free(transposed);
    }

    free(local_rows);
    free(local_transposed);
    MPI_Finalize();
    return 0;
}
