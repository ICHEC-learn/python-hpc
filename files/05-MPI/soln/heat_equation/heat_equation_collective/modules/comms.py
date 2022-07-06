
from mpi4py import MPI
from mpi4py.MPI import Request
import numpy

def halo_exchange(field, rank, num_processes):
    # determine neighbours
    up = (rank - 1) % num_processes
    down = (rank + 1) % num_processes
    
    # send bottom border down
    requests = [MPI.COMM_WORLD.Isend(field[-2], down)]
    # send top border up
    requests.append(MPI.COMM_WORLD.Isend(field[1], up))
    # receive down neighbour's border to down halo region
    requests.append(MPI.COMM_WORLD.Irecv(field[-1], down))
    # receive up neighbour's border to up halo region
    requests.append(MPI.COMM_WORLD.Irecv(field[0], up))
    
    return requests


def distribute_subfields(field, rank, num_processes):
    dims = numpy.empty(2, dtype=int)
    if rank == 0:
        num_rows, num_cols = field.shape

        num_subfield_rows = num_rows // num_processes
        dims[0] = num_subfield_rows + 2
        dims[1] = num_cols

    MPI.COMM_WORLD.Bcast(dims, 0)
    local_field = numpy.zeros(dims)
    MPI.COMM_WORLD.Scatter(field, local_field[1:-1], 0)

    return local_field


def gather_subfields(local_field, rank, num_processes):
    dims = numpy.empty(2, dtype=int)
    dims[0], dims[1] = local_field[1:-1, ].shape
    dims[0] *= num_processes
    field = numpy.empty(dims)
    MPI.COMM_WORLD.Gather(local_field[1:-1, ], field, 0)

    return field
