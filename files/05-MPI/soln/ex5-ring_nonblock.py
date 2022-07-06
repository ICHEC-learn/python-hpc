from mpi4py import MPI
import numpy as np

rcv_buf = np.empty((), dtype=np.intc)
status = MPI.Status()

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
size = comm.Get_size()

right = (my_rank+1) % size
left  = (my_rank-1+size) % size

sum = 0
snd_buf = np.array(my_rank, dtype=np.intc) # 0 dimensional integer array with 1 element initialized with the value of my_rank

for i in range(size):   
   # Official solution to be used in your code outside of lesson
   request = comm.Isend((snd_buf, 1, MPI.INT), dest=right, tag=17)
   comm.Recv((rcv_buf, 1, MPI.INT), source=left,  tag=17, status=status)
   request.Wait(status)
   np.copyto(snd_buf, rcv_buf) 
   sum += rcv_buf
print(f"PE{my_rank}:\tSum = {sum}")