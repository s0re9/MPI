#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10         // Grid size (NxN)
#define MAX_ITER 1000   // Max iterations
#define THRESHOLD 0.001 // Convergence threshold

// Initialize grid with boundary conditions
void initialize_grid(double** grid) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = (i == 0 || i == N - 1 || j == 0 || j == N - 1) ? 100.0 : 0.0;
        }
    }
}

// Print the grid
void print_grid(double** grid) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", grid[i][j]);
        }
        printf("\n");
    }
}

// Compute new value based on neighbors
double compute_new_value(double top, double bottom, double left, double right) {
    return 0.25 * (top + bottom + left + right);
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            printf("Error: Grid size N must be divisible by the number of processes!\n");
        }
        MPI_Finalize();
        return -1;
    }

    int rows_per_proc = N / size;

    // **Allocate dynamic memory for local grid**
    double** local_grid = (double**)malloc((rows_per_proc + 2) * sizeof(double*));
    double** new_local_grid = (double**)malloc((rows_per_proc + 2) * sizeof(double*));
    for (int i = 0; i < rows_per_proc + 2; i++) {
        local_grid[i] = (double*)malloc(N * sizeof(double));
        new_local_grid[i] = (double*)malloc(N * sizeof(double));
    }

    // **Root initializes full grid**
    double** full_grid = NULL;
    if (rank == 0) {
        full_grid = (double**)malloc(N * sizeof(double*));
        for (int i = 0; i < N; i++) {
            full_grid[i] = (double*)malloc(N * sizeof(double));
        }
        initialize_grid(full_grid);
    }

    // **Scatter grid rows**
    for (int i = 1; i <= rows_per_proc; i++) {
        MPI_Scatter(&full_grid[i - 1][0], N, MPI_DOUBLE,
            &local_grid[i][0], N, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
    }

    // **Set fixed boundary conditions**
    if (rank == 0) { // Top boundary
        for (int j = 0; j < N; j++) {
            local_grid[0][j] = 100.0;
        }
    }
    if (rank == size - 1) { // Bottom boundary
        for (int j = 0; j < N; j++) {
            local_grid[rows_per_proc + 1][j] = 100.0;
        }
    }

    int iter = 0;
    double diff;
    do {
        // **Exchange boundary rows**
        if (rank > 0) {
            MPI_Sendrecv(local_grid[1], N, MPI_DOUBLE, rank - 1, 0,
                local_grid[0], N, MPI_DOUBLE, rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(local_grid[rows_per_proc], N, MPI_DOUBLE, rank + 1, 0,
                local_grid[rows_per_proc + 1], N, MPI_DOUBLE, rank + 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // **Compute new values**
        diff = 0.0;
        for (int i = 1; i <= rows_per_proc; i++) {
            for (int j = 1; j < N - 1; j++) {
                new_local_grid[i][j] = compute_new_value(
                    local_grid[i - 1][j], local_grid[i + 1][j],
                    local_grid[i][j - 1], local_grid[i][j + 1]);
                diff = fmax(diff, fabs(new_local_grid[i][j] - local_grid[i][j]));
            }
        }

        // **Copy new values**
        for (int i = 1; i <= rows_per_proc; i++) {
            for (int j = 1; j < N - 1; j++) {
                local_grid[i][j] = new_local_grid[i][j];
            }
        }

        // **Check for convergence**
        double global_diff;
        MPI_Allreduce(&diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (global_diff < THRESHOLD) {
            break;
        }

        iter++;
    } while (iter < MAX_ITER);

    // **Gather final grid at root**
    for (int i = 1; i <= rows_per_proc; i++) {
        MPI_Gather(&local_grid[i][0], N, MPI_DOUBLE,
            &full_grid[i - 1][0], N, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
    }

    // **Print final result**
    if (rank == 0) {
        printf("\nFinal Heat Distribution:\n");
        print_grid(full_grid);
    }

    // **Free allocated memory**
    for (int i = 0; i < rows_per_proc + 2; i++) {
        free(local_grid[i]);
        free(new_local_grid[i]);
    }
    free(local_grid);
    free(new_local_grid);

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            free(full_grid[i]);
        }
        free(full_grid);
    }

    MPI_Finalize();
    return 0;
}
