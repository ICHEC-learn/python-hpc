from mpi4py import MPI

comm = MPI.COMM_WORLD 
rank = comm.Get_rank()
size = comm.Get_size()

left_rank = (rank - 1) % size
right_rank = (rank + 1) % size

if (rank % 2) == 0:
    comm.send(rank, dest=right_rank)
    comm.send(rank, dest=left_rank)
    left = comm.recv(source=left_rank)
    right = comm.recv(source=right_rank)
else:
    left = comm.recv(source=left_rank)
    right = comm.recv(source=right_rank)
    comm.send(rank, dest=right_rank)
    comm.send(rank, dest=left_rank)   

print ('I am process ', rank, ', my neighbours are processes', left, ' and ', right)