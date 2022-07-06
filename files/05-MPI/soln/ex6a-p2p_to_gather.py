from mpi4py import MPI
import numpy as np

#MPI-related data
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_procs = comm.Get_size()

result = np.array(my_rank**2, dtype=np.double)
print(f"I am process {my_rank} out of {num_procs}, result={result:f}")

if (my_rank == 0):
   res_arr = np.empty(num_procs, dtype=np.double)
else:
   res_arr = None

comm.Gather((result,1,MPI.DOUBLE), (res_arr,1,MPI.DOUBLE), root=0) 
if (my_rank == 0):
   for rank in range(num_procs):
      print(f"I'm proc 0: result of process {rank} is {res_arr[rank]:f}")