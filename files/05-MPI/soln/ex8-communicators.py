from mpi4py import MPI
import numpy as np

sumA = np.empty((), dtype=np.intc)
sumB = np.empty((), dtype=np.intc)

comm_world = MPI.COMM_WORLD
world_size = comm_world.Get_size()
my_world_rank = np.array(comm_world.Get_rank(), dtype=np.intc)

mycolor = (my_world_rank > (world_size-1)//3)
# This definition of mycolor implies that the first color is 0
sub_comm = comm_world.Split(mycolor, 0)
sub_size = sub_comm.Get_size()
my_sub_rank = np.array(sub_comm.Get_rank(), dtype=np.intc)

# Compute sum of all ranks.
sub_comm.Allreduce(my_world_rank, (sumA, 1, MPI.INT), MPI.SUM) 
sub_comm.Allreduce(my_sub_rank,   (sumB, 1, MPI.INT), MPI.SUM)

print("PE world:{:3d}, color={:d} sub:{:3d}, SumA={:3d}, SumB={:3d} in sub_comm".format( 
          my_world_rank, mycolor, my_sub_rank, sumA, sumB))