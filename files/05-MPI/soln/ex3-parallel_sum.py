from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 100
local_n = n // size

local_data = np.empty(local_n , dtype=int)

# distribute data
if rank==0:
    data = np.arange(100, dtype=int)
    local_data = data[:local_n] # choose appropriate range

    # communicate
    for i in range(1, size):
        comm.Send(data[i*local_n:(i + 1)*local_n], i)
else:    
    comm.Recv(local_data, 0)

# compute local sum
local_sum = np.sum(local_data)
partial_sum = {'sum': local_sum}

# collect data
if rank==0:
    partial_sums = np.empty(size, dtype=int)
    partial_sums[0] = local_sum

    # communicate
    for i in range(1, size):
        partial_sums[i] = comm.recv(source=i)['sum']

    # calculate total sum
    total_sum = np.sum(partial_sums)

    # print result
    print('Total sum: ', total_sum)
else:
    comm.send(partial_sum, 0)
