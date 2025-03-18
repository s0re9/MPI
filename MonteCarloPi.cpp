#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size, i, local_count = 0, total_count;
    long long int n = 1000000; // Number of points per process

    MPI_Init(&argc, &argv);  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    srand(time(NULL) + rank);  // Seed random number generator for each process

    // Monte Carlo Simulation
    for (i = 0; i < n; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    // Gather results from all processes to process 0
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double pi = 4.0 * (double)total_count / (n * size);
        printf("Estimated Pi: %lf\n", pi);
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}
