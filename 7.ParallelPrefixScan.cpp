#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8  // Array size (should be divisible by number of processes)

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements_per_proc = N / size;  // Divide workload
    int* data = NULL;
    int* local_data = (int*)malloc(elements_per_proc * sizeof(int));
    int* local_prefix = (int*)malloc(elements_per_proc * sizeof(int));

    if (rank == 0) {
        // **Master process initializes data**
        data = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) {
            data[i] = i + 1;  // Example: {1, 2, 3, 4, 5, 6, 7, 8}
        }
    }

    // **Scatter data among processes**
    MPI_Scatter(data, elements_per_proc, MPI_INT, local_data, elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // **Compute local prefix sum**
    local_prefix[0] = local_data[0];
    for (int i = 1; i < elements_per_proc; i++) {
        local_prefix[i] = local_prefix[i - 1] + local_data[i];
    }

    int offset = 0;
    MPI_Exscan(&local_prefix[elements_per_proc - 1], &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // **Adjust local prefix sum based on offset**
    if (rank != 0) {
        for (int i = 0; i < elements_per_proc; i++) {
            local_prefix[i] += offset;
        }
    }

    // **Gather results at Rank 0**
    MPI_Gather(local_prefix, elements_per_proc, MPI_INT, data, elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // **Print final result at Rank 0**
    if (rank == 0) {
        printf("Prefix Sum: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
        free(data);
    }

    free(local_data);
    free(local_prefix);
    MPI_Finalize();
    return 0;
}
