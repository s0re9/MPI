#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

// Function to check if a number is prime
bool is_prime(int n) {
    if (n < 2) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    int rank, size;
    const int max_value = 20;  // Find primes up to this number

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            cout << "Error: Please run with at least 2 processes (1 master, 1 worker)." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) 
    { 
        // MASTER PROCESS
        int num = 2;
        int active_workers = size - 1;
        vector<int> primes;

        cout << "Master started with " << active_workers << " workers." << endl;

        while (active_workers > 0) 
        {
            int received_value, worker_rank;
            MPI_Status status;

            // Receive a message from any worker
            MPI_Recv(&received_value, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            worker_rank = status.MPI_SOURCE;

            // Store only positive numbers as primes
            if (received_value > 0) 
            {  
                primes.push_back(received_value);
            }

            // Send the next number or terminate the worker
            if (num <= max_value) 
            {
                MPI_Send(&num, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
                num++;
            } 
            else 
            {
                int terminate_signal = -1;
                MPI_Send(&terminate_signal, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        // Print all primes found
        cout << "Primes up to " << max_value << ": ";
        for (int prime : primes) 
        {
            cout << prime << " ";
        }
        cout << endl;
    } 
    else 
    { 
        // WORKER PROCESS
        while (true) 
        {
            int request_signal = 0;
            MPI_Send(&request_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            int num_to_test;
            MPI_Recv(&num_to_test, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (num_to_test == -1) 
            {
                break;  // Termination signal received
            }

            int result = is_prime(num_to_test) ? num_to_test : -num_to_test;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
