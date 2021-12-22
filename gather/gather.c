#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
int N = 4;

int get_row(int rank) {
    return rank / N;
}

int get_col(int rank) {
    return rank % N;
}

int main(int argc, char **argv) {
    int rank, world_size, idx;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int row = get_row(rank);
    int col = get_col(rank);
    MPI_Status tag;
    MPI_Request req;

    if (row == 0 && col == 0) {
        // (0, 0)
        MPI_Status tags[world_size - 1];
        MPI_Request reqs[world_size - 1];
        int *values = malloc(world_size * sizeof(int));
        values[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            if (get_col(i) == 0) {
                MPI_Irecv(&values[i], 1, MPI_INT, N, i, MPI_COMM_WORLD, &reqs[i - 1]);
            } else {
                MPI_Irecv(&values[i], 1, MPI_INT, 1, i, MPI_COMM_WORLD, &reqs[i - 1]);
            }
        }
        MPI_Waitall(world_size - 1, reqs, tags);
        printf("printing accumulated array: \n");
        for (int i = 0; i < world_size; ++i) {
            if (i != values[i]) {
                printf("Wrong answer from process %d", i);
                break;
            }
            printf("%d ", values[i]);
        }
        printf("\n");
    } else if (row == 0) {
        // (0, any)
        int buf_size = (N - col) * N - 1;
        MPI_Request reqs[buf_size];
        int *buf = malloc(buf_size * sizeof(int));
        for (int i = 0; i < N - 1; i++) {
            MPI_Irecv(&buf[i], 1, MPI_INT, rank + N, rank + N * i + N, MPI_COMM_WORLD, &reqs[i]);
        }
        for (int i = 1; i < N - col; ++i) {
            for (int j = 0; j < N; ++j) {
                MPI_Irecv(&buf[i * N + j - 1], 1, MPI_INT, rank + 1, rank + j * N + i, MPI_COMM_WORLD, &reqs[i * N + j - 1]);
            }
        }
        MPI_Isend(&rank, 1, MPI_INT, rank - 1, rank, MPI_COMM_WORLD, &req); 
        for (int i = 0; i < buf_size; ++i) {
            MPI_Waitany(buf_size, reqs, &idx, &tag);
            MPI_Isend(&buf[idx], 1, MPI_INT, rank - 1, tag.MPI_TAG, MPI_COMM_WORLD, &req); 
        }
    } else if (row == N - 1) {
        // (N - 1, any)
        MPI_Isend(&rank, 1, MPI_INT, rank - N, rank, MPI_COMM_WORLD, &req); 
    } else {
        int buf_size = N - row - 1;
        MPI_Request reqs[buf_size];
        int *buf = malloc(buf_size * sizeof(int));
        for (int i = 0; i < N - row - 1; i++) {
            MPI_Irecv(&buf[i], 1, MPI_INT, rank + N, rank + N * (i + 1), MPI_COMM_WORLD, &reqs[i]);
        }
        MPI_Isend(&rank, 1, MPI_INT, rank - N, rank, MPI_COMM_WORLD, &req); 
        for (int i = 0; i < buf_size; ++i) {
            MPI_Waitany(buf_size, reqs, &idx, &tag);
            MPI_Isend(&buf[idx], 1, MPI_INT, rank - N, tag.MPI_TAG, MPI_COMM_WORLD, &req); 
        }
    }
    MPI_Finalize();
    return 0;
}