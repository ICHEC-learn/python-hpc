from mpi4py import MPI

comm = MPI.COMM_WORLD 
rank = comm.Get_rank()
size = comm.Get_size()

left_rank = (rank - 1) % size
right_rank = (rank + 1) % size

right = comm.sendrecv(rank, dest=right_rank, source=left_rank)
left = comm.sendrecv(rank, dest=left_rank, source=right_rank)

print ('I am process ', rank, ', my neighbours are processes', left, ' and ', right)