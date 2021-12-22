#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <mpi.h>
#include <mpi-ext.h>


#define Max(a, b) ((a) > (b) ? (a) : (b))
#define m_printf if (proc == 0) printf
enum Work_State {
    BACKUP,
    
    RESTORE
};
enum Work_State Current_State = BACKUP;
int *prev = NULL;
int *next = NULL;
int msg_list_len = 0;
int proc, n_proc;
int idle = false;
double *tmp = NULL;
double ***A = NULL;
double ***B = NULL;
int CHECK_FREQ = 10;
int fst_layer, lst_layer, lst_proc;
char path[256];
MPI_Comm work_comm;
long long N;
int n_iters, n_tries;
int n_layers;
int rank;


static void errhandler(MPI_Comm* pcomm, int* perr, ...) {
    MPI_Comm comm = *pcomm;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int i, rank, size, nf, len, eclass;
    MPI_Group group_c, group_f;
    int *ranks_gc, *ranks_gf;

    MPI_Error_class(err, &eclass);
    if (MPIX_ERR_PROC_FAILED != eclass) {
        MPI_Abort(comm, err);
    }

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPIX_Comm_failure_ack(comm);
    MPIX_Comm_failure_get_acked(comm, &group_f);
    MPI_Group_size(group_f, &nf);
    MPI_Error_string(err, errstr, &len);
    printf("Rank %d / %d: Notified of error %s. %d found dead: ", rank, size, errstr, nf);

    ranks_gf = (int*) malloc(nf * sizeof(int));
    ranks_gc = (int*) malloc(nf * sizeof(int));
    MPI_Comm_group(comm, &group_c);
    for (i = 0; i < nf; i++) {
        ranks_gf[i] = i;
    }
    MPI_Group_translate_ranks(group_f, nf, ranks_gf, group_c, ranks_gc);
    for (i = 0; i < nf; i++) {
        printf("%d ", ranks_gc[i]);
    }
    printf("\n");


    for (int i = 0; i < msg_list_len; ++i) {
        if (next[i] == ranks_gc[0]) {
            next[i] = n_proc - 1;
        } else if (next[i] > ranks_gc[0]) {
            next[i] = next[i] - 1;
        }
        if (prev[i] == ranks_gc[0]) {
            prev[i] = n_proc - 1;
        } else if (prev[i] > ranks_gc[0]) {
            prev[i] = prev[i] - 1;
        }
    }

    if (idle) {
        idle = false;
        proc = ranks_gc[0];

        if (N < n_proc) {
            if (proc < N) {
                fst_layer = proc;
                lst_layer = proc + 1;
                lst_proc = N - 1;
            } else {
                fst_layer = 0;
                lst_layer = 0;
                lst_proc = 0;
            }
        } else {
            fst_layer = N * proc / n_proc;
            lst_layer = N * (proc + 1) / n_proc;
            lst_proc = n_proc - 1;
        }
        n_layers = lst_layer - fst_layer;

        tmp = malloc((n_layers + 2) * N * N * sizeof(*tmp));
        A = malloc((n_layers + 2) * sizeof(*A));
        for (int i = 0; i < n_layers + 2; ++i) {
            A[i] = malloc(N * sizeof(*A[i]));
            
            for (int j = 0; j < N; ++j) {
                A[i][j] = &tmp[i * N * N + j * N];
            }
        }

        if (proc == 0) {
            tmp = malloc(N * N * N * sizeof(*tmp));
            B = malloc(N * sizeof(*B));
            for (int i = 0; i < N; ++i) {
                B[i] = malloc(N * sizeof(*B[i]));
                
                for (int j = 0; j < N; ++j) {
                    B[i][j] = &tmp[i * N * N + j * N];
                }
            }
        }
    }

    MPI_Comm tmp_comm;
    MPIX_Comm_shrink(work_comm, &tmp_comm);

    MPI_Comm_free(&work_comm);
    work_comm = tmp_comm;

    Current_State = RESTORE;
}

void dump_data(const char *save_path, double ***arr, long long N, int n_layers) {
    FILE *f;
    f = fopen(save_path, "w");

    for (int i = 0; i < n_layers + 2; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                fprintf(f, "%lf ", arr[i][j][k]);
            }
            fprintf(f, "\n");
        }
    }
    fclose(f);

}

void load_data(const char *read_path, double ***arr, long long N, int n_layers)
{
    FILE *f;
    f = fopen(read_path, "r");

    for (int i = 0; i < n_layers + 2; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double buf;
                fscanf(f, "%lf ", &buf);
                arr[i][j][k] = buf;
            }
        }
    }
    fclose(f);

}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);


    MPI_Errhandler handler;
    MPI_Comm_create_errhandler(&errhandler, &handler);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, handler);


    rank = proc;


    msg_list_len = n_proc;
    prev = malloc(msg_list_len * sizeof(prev));
    next = malloc(msg_list_len * sizeof(next));
    for (int i = 0; i < msg_list_len; ++i) {
        prev[i] = i - 1;
        next[i] = i + 1;
    }

    int victim = -1;

    sscanf(argv[1], "%lld", &N);
    sscanf(argv[2], "%d", &n_iters);

    if (argc == 4) {
        sscanf(argv[3], "%d", &victim);
    }

    double w = 0.5;
    double eps;

    
    MPI_Request req[4];
    MPI_Status status[4];

    n_proc = n_proc - 1;
    idle = proc == n_proc;

    int color;
    if (!idle) {
        if (N < n_proc) {
            if (proc < N) {
                fst_layer = proc;
                lst_layer = proc + 1;
                lst_proc = N - 1;
            } else {
                fst_layer = 0;
                lst_layer = 0;
                lst_proc = 0;
            }
        } else {
            fst_layer = N * proc / n_proc;
            lst_layer = N * (proc + 1) / n_proc;
            lst_proc = n_proc - 1;
        }
        n_layers = lst_layer - fst_layer;
        color = !n_layers;
    } else {
        color = 0;
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, proc, &work_comm);
    if (!idle) {
        tmp = malloc((n_layers + 2) * N * N * sizeof(*tmp));
        A = malloc((n_layers + 2) * sizeof(*A));
        for (int i = 0; i < n_layers + 2; ++i) {
            A[i] = malloc(N * sizeof(*A[i]));
            
            for (int j = 0; j < N; ++j) {
                A[i][j] = &tmp[i * N * N + j * N];
            }
        }

        if (proc == 0) {
            tmp = malloc(N * N * N * sizeof(*tmp));
            B = malloc(N * sizeof(*B));
            for (int i = 0; i < N; ++i) {
                B[i] = malloc(N * sizeof(*B[i]));
                
                for (int j = 0; j < N; ++j) {
                    B[i][j] = &tmp[i * N * N + j * N];
                }
            }
        }
        for (int i = 0; i < n_layers + 2; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    if (fst_layer + i == 0 || fst_layer + i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1) {
                        A[i][j][k]= 0.;
                    } else {
                        A[i][j][k] = (4. + fst_layer + i + j + k);
                    }
                }
            }
        }
    }

    double start = MPI_Wtime();

    int it = 0;
    for (; it < n_iters; ++it) {
        if (it % CHECK_FREQ == 0) {
            if (!idle) {
                sprintf(path, "./checkpoints/%d.txt", proc);
                switch (Current_State) {
                    case BACKUP:
                        dump_data(path, A, N, n_layers);
                        break;
                    case RESTORE:
                        load_data(path, A, N, n_layers);
                        it -= CHECK_FREQ;
                        Current_State = BACKUP;
                        break;
                }
            }
            MPI_Barrier(work_comm);
            if (rank == victim) {
                raise(SIGKILL);
            }
            MPI_Barrier(work_comm);
        }
        if (idle) {
            MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_MAX, work_comm);
            continue;
        }

        eps = 0.;

        for (int i = 1; i < n_layers + 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                for (int k = 1 + (i + j) % 2; k < N - 1; k += 2) {
                    double b;
                    b = w * ( (A[i - 1][j][k] + A[i + 1][j][k]
                                + A[i][j - 1][k] + A[i][j + 1][k]
                                + A[i][j][k - 1] + A[i][j][k + 1]
                            ) / 6. - A[i][j][k]);
                    eps = Max(fabs(b), eps);
                    A[i][j][k] = A[i][j][k] + b;
                }
            }
        }

        for (int i = 1; i < n_layers + 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                for (int k = 1 + (i + j + 1) % 2; k < N - 1; k += 2) {
                    double b;
                    b = w * ( (A[i - 1][j][k] + A[i + 1][j][k]
                                + A[i][j - 1][k] + A[i][j + 1][k]
                                + A[i][j][k - 1] + A[i][j][k + 1]
                            ) / 6. - A[i][j][k]);
                    A[i][j][k] = A[i][j][k] + b;
                }
            }
        }

        if (proc != 0) {
            MPI_Irecv(A[0][0], N * N, MPI_DOUBLE, prev[proc], 217,
                work_comm, &req[0]);
        }
        if (proc != lst_proc) {
            MPI_Isend(A[n_layers][0], N * N, MPI_DOUBLE, next[proc], 217,
                work_comm, &req[2]);
        }
        if (proc != lst_proc) {
            MPI_Irecv(A[n_layers + 1][0], N * N, MPI_DOUBLE, next[proc], 556,
                work_comm, &req[3]);
        }
        if (proc != 0) {
            MPI_Isend(A[1][0], N * N, MPI_DOUBLE, prev[proc], 556,
                work_comm, &req[1]);
        }
        int ll = 4;
        int shift = 0;
        if (proc == 0) {
            ll = 2;
            shift = 2;
        }
        if (proc == lst_proc) {
            ll = 2;
        }
        MPI_Waitall(ll, &req[shift], &status[0]);
        MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_MAX, work_comm);
    }

    double end = MPI_Wtime();

    if (!idle) {
        if (proc == 0) {
            for (int i = 0; i < n_layers; ++i) {
                for (int j = 0; j < N; ++j) {
                    for (int k = 0; k < N; ++k) {
                        B[i][j][k] = A[i][j][k];
                    }
                }
            }

            for (int i = 1; i <= lst_proc; i++) {
                int recv_layer = 0;
                int recv_layer_end = 0;
                int recv_layer_count = 0;
                if (N < n_proc) {
                    recv_layer = i;
                    recv_layer = i + 1;
                } else {
                    recv_layer = N * i / n_proc;
                    recv_layer_end = N * (i + 1) / n_proc;
                }

                recv_layer_count = recv_layer_end - recv_layer;
                MPI_Recv(B[recv_layer][0], recv_layer_count * N * N, MPI_DOUBLE, next[i-1],
                    1337, work_comm, MPI_STATUS_IGNORE);
            }

            double s = 0.;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    for (int k = 0; k < N; ++k) {
                        s = s + B[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
                    }
                }
            }
            m_printf("S = %f\niters = %d\n", s, it);
        } else if (proc == lst_proc) {
            MPI_Send(A[0][0], n_layers * N * N, MPI_DOUBLE, prev[1], 1337, work_comm);
        } else {
            MPI_Send(A[1][0], n_layers * N * N, MPI_DOUBLE, prev[1], 1337, work_comm);
        }
    }

    m_printf("Time elapsed: %lf s\n", end - start);
    if (!idle) {
        free(A[0][0]);
        for (int i = 0; i < n_layers + 2; ++i) {
            free(A[i]);
        }
        free(A);

        if (proc == 0) {
            free(B[0][0]);
            for (int i = 0; i < N; ++i) {
                free(B[i]);
            }
            free(B);
        }
    }

    MPI_Finalize();

    return 0;
}
