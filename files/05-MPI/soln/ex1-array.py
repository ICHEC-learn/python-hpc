from mpi4py import MPI

# communicator containing all processes
comm = MPI.COMM_WORLD 

size = comm.Get_size()
rank = comm.Get_rank()

arr = [1,2,3,4]
for i in range(len(arr)):
    arr[i] = i*rank


print("I am rank %d in group of %d processes" % (rank, size))
print("My list is", arr)