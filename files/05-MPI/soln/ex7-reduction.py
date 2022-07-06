from mpi4py import MPI
import numpy as np

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

snd_buf = np.array(my_rank, dtype=np.intc)
sum = np.empty((), dtype=np.intc)

# Compute sum of all ranks.
comm_world.Allreduce(snd_buf, (sum,1,MPI.INT), op=MPI.SUM )
# Also possible
# comm_world.Allreduce((snd_buf,1,MPI.INT), (sum,1,MPI.INT), op=MPI.SUM)
# Shortest version in python is
# comm_world.Allreduce(snd_buf, sum)

print(f"PE{my_rank}:\tSum = {sum}")