from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 100
local_n = n // size
local_data = np.empty(local_n , dtype=int)

if rank==0:
    data = np.arange(100, dtype=int)
else:
    data = None

# distribute data
comm.Scatter(data, local_data, 0)

# compute local sum
local_sum = np.sum(local_data)
partial_sums = np.empty(size, dtype=int)

# collect data
comm.Gather(local_sum, partial_sums)
if rank==0:
    # calculate total sum
    total_sum = np.sum(partial_sums)

    # print result
    print('Total sum: ', total_sum)