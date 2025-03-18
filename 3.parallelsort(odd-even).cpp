#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 16  // Total number of elements (change as needed)

// Function to swap two elements
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Odd-Even Sort for local array
void odd_even_sort(int* arr, int n) {
    int sorted = 0;
    while (!sorted) {
        sorted = 1;

        // Odd phase
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(&arr[i], &arr[i + 1]);
                sorted = 0;
            }
        }

        // Even phase
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(&arr[i], &arr[i + 1]);
                sorted = 0;
            }
        }
    }
}

// Print array
void print_array(int* arr, int n, const char* msg) {
    printf("%s: ", msg);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    int rank, size;
    int* arr = NULL;  // Full array (only in root process)
    int* local_arr;   // Local chunk of array
    int n_per_proc;   // Elements per process

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure SIZE is divisible by number of processes
    if (SIZE % size != 0) {
        if (rank == 0)
            printf("Error: SIZE must be divisible by number of processes.\n");
        MPI_Finalize();
        return -1;
    }

    n_per_proc = SIZE / size;
    local_arr = new int[n_per_proc];  // Use dynamic memory allocation

    // Root initializes array
    if (rank == 0) {
        arr = new int[SIZE];
        srand(static_cast<unsigned int>(time(NULL)));  // FIXED: Explicit cast

        for (int i = 0; i < SIZE; i++) {
            arr[i] = rand() % 100;  // Random numbers 0-99
        }
        print_array(arr, SIZE, "Unsorted Array");
    }

    // Scatter array to all processes
    MPI_Scatter(arr, n_per_proc, MPI_INT, local_arr, n_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Local sorting
    odd_even_sort(local_arr, n_per_proc);

    // Odd-Even Transposition
    for (int phase = 0; phase < size; phase++) {
        int partner;
        if (phase % 2 == 0) {
            partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
        }
        else {
            partner = (rank % 2 == 0) ? rank - 1 : rank + 1;
        }

        if (partner >= 0 && partner < size) {
            int* neighbor_data = new int[n_per_proc];  // FIXED: Use heap memory

            MPI_Sendrecv(local_arr, n_per_proc, MPI_INT, partner, 0,
                neighbor_data, n_per_proc, MPI_INT, partner, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Merge two chunks
            int* merged = new int[2 * n_per_proc];  // FIXED: Use heap allocation
            for (int i = 0; i < n_per_proc; i++) merged[i] = local_arr[i];
            for (int i = 0; i < n_per_proc; i++) merged[n_per_proc + i] = neighbor_data[i];

            // Sort merged data
            odd_even_sort(merged, 2 * n_per_proc);

            // Keep only the relevant half
            if (rank < partner) {
                for (int i = 0; i < n_per_proc; i++) local_arr[i] = merged[i];
            }
            else {
                for (int i = 0; i < n_per_proc; i++) local_arr[i] = merged[n_per_proc + i];
            }

            delete[] neighbor_data;
            delete[] merged;
        }
    }

    // Gather sorted sub-arrays
    MPI_Gather(local_arr, n_per_proc, MPI_INT, arr, n_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process prints sorted array
    if (rank == 0) {
        print_array(arr, SIZE, "Sorted Array");
        delete[] arr;
    }

    delete[] local_arr;
    MPI_Finalize();
    return 0;
}
