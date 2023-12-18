from mpi4py import MPI
from numpy import empty, array, int32, float64, linspace, hstack

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

# функция задания начального условия
def u_init(x):
    return 11 - h * (x - 1)

# функция задания левого граничного условия
def u_left(x):
    return 11


if rank == 0:
    start_time = MPI.Wtime()

a, b = (0, 1)
t_0, T = (0, 1)

N, M = (10, 100)
x, h = linspace(a, b, N + 1, retstep=True)
t, tau = linspace(t_0, T, M + 1, retstep=True)

if rank == 0:
    ave, res = divmod(N + 1, numprocs)
    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)
    for k in range(0, numprocs):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        if k == 0:
            displs[k] = 0
        else:
            displs[k] = displs[k - 1] + rcounts[k - 1]
else:
    rcounts, displs = None, None

N_part = array(0, dtype=int32)
comm.Scatter([rcounts, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)

if rank == 0:
    rcounts_from_0 = empty(numprocs, dtype=int32)
    displs_from_0 = empty(numprocs, dtype=int32)
    rcounts_from_0[0] = rcounts[0] + 1
    displs_from_0[0] = 0
    for k in range(1, numprocs - 1):
        rcounts_from_0[k] = rcounts[k] + 2
        displs_from_0[k] = displs[k] - 1
    rcounts_from_0[numprocs - 1] = rcounts[numprocs - 1] + 1
    displs_from_0[numprocs - 1] = displs[numprocs - 1] - 1
else:
    rcounts_from_0 = None;
    displs_from_0 = None

N_part_aux = array(0, dtype=int32)
comm.Scatter([rcounts_from_0, 1, MPI.INT], [N_part_aux, 1, MPI.INT], root=0)

if rank == 0:
    u = empty((M + 1, N + 1), dtype=float64)
    for n in range(N + 1):
        u[0, n] = u_init(x[n])
else:  # rank != 0
    u = empty((M + 1, 0), dtype=float64)

u_part = empty(N_part, dtype=float64)
u_part_aux = empty(N_part_aux, dtype=float64)

for m in range(M):
    comm.Scatterv([u[m], rcounts_from_0, displs_from_0, MPI.DOUBLE], [u_part_aux, N_part_aux, MPI.DOUBLE], root=0)

    for n in range(1, N_part_aux - 1):
        u_part[n - 1] = u_part_aux[n] + tau / h * (u_part_aux[n - 1] - u_part_aux[n]) - tau * (3 * h * (n - 1))

    if rank == 0:
        u_part = hstack((array(u_left(t[m + 1]), dtype=float64), u_part[0:N_part - 1]))

    comm.Gatherv([u_part, N_part, MPI.DOUBLE], [u[m + 1], rcounts, displs, MPI.DOUBLE], root=0)

if rank == 0:
    end_time = MPI.Wtime()
    print(f"Elapsed time is {end_time - start_time:.4f} sec.")
    print(f"Number of MPI processes is {numprocs}")
    import pandas as pd
    from tabulate import tabulate
    df = pd.DataFrame(u)
    xsteps = ['x = ' + str(round(i, 2)) for i in x]
    print(tabulate(u, headers=xsteps, tablefmt='psql', showindex=('t = ' + str(round(i, 2)) for i in t)))
