#include <mpi.h>
#include <stdio.h>

static long num_steps = 100000;
double step;

int main(int argc, char* argv[]) {
    int rank, size, i;
    double x, pi, sum = 0.0, local_sum = 0.0;

    MPI_Init(&argc, &argv);                     // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // Get number of processes

    // Broadcast num_steps from process 0 to all processes
    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    step = 1.0 / (double)num_steps;

    // Each process computes its part of the sum
    for (i = rank; i < num_steps; i += size) {
        x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    // Reduce all partial sums into process 0
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Process 0 computes the final value of π
    if (rank == 0) {
        pi = step * sum;
        printf("Calculated π = %.15f\n", pi);
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}
