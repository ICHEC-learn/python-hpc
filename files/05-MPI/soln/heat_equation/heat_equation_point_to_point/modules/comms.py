
from mpi4py import MPI
import numpy

def halo_exchange(field, rank, num_processes):
    # determine neighbours
    up = (rank - 1) % num_processes
    down = (rank + 1) % num_processes
    
    # checkerboard scheme to avoid deadlocks
    if (rank % 2) == 0:
        # send bottom border down
        MPI.COMM_WORLD.Send(field[-2], down)
        # send top border up
        MPI.COMM_WORLD.Send(field[1], up)
        # receive down neighbour's border to down halo region
        MPI.COMM_WORLD.Recv(field[-1], down)
        # receive up neighbour's border to up halo region
        MPI.COMM_WORLD.Recv(field[0], up)
    else:
        # match alternating communications
        MPI.COMM_WORLD.Recv(field[0], up)
        MPI.COMM_WORLD.Recv(field[-1], down)
        MPI.COMM_WORLD.Send(field[1], up)
        MPI.COMM_WORLD.Send(field[-2], down)
        

def distribute_subfields(field, rank, num_processes):
    dims = numpy.empty(2, dtype=int)
    if rank == 0:
        num_rows, num_cols = field.shape

        num_subfield_rows = num_rows // num_processes
        dims[0] = num_subfield_rows + 2 # add halo regions
        dims[1] = num_cols
        
        for i in range(1, num_processes):
            MPI.COMM_WORLD.Send(dims, i)
    else:
        MPI.COMM_WORLD.Recv(dims, 0)

    local_field = numpy.zeros(dims)
    if rank == 0:
        local_field[1:-1] = field[0:(dims[0] - 2)]
        for i in range(1, num_processes):
            MPI.COMM_WORLD.Send(field[i*(dims[0] - 2):(i + 1)*(dims[0] - 2)], i)
    else:
        MPI.COMM_WORLD.Recv(local_field[1:-1], 0)

    return local_field


def gather_subfields(local_field, rank, num_processes):
    dims = numpy.empty(2, dtype=int)
    dims[0], dims[1] = local_field[1:-1].shape
    dims[0] *= num_processes
    field = numpy.empty(dims)
    
    if rank == 0:
        field[:dims[0]//num_processes] = local_field[1:-1]
        for i in range(1, num_processes):
            MPI.COMM_WORLD.Recv(field[i*dims[0]//num_processes:(i + 1)*dims[0]//num_processes], i)
    else:
        MPI.COMM_WORLD.Send(local_field[1:-1], 0)

    return field
