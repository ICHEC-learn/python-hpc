
from mpi4py.MPI import Request

def evolve(u, u_previous, a, dt, dx2, dy2, requests):
    # evolve non-boundary regions
    del_sqrd_u = (u_previous[1:-3, 1:-1] - 2 * u_previous[2:-2, 1:-1] +
                  u_previous[3:-1, 1:-1]) / dx2 + (u_previous[2:-2, :-2] -
                                                   2 * u_previous[2:-2, 1:-1] +
                                                   u_previous[2:-2, 2:]) / dy2

    u[2:-2, 1:-1] = u_previous[2:-2, 1:-1] + dt * a * del_sqrd_u

    Request.waitall(requests)

    # evolve boundary regions
    del_sqrd_u_up = (u_previous[0, 1:-1] - 2 * u_previous[1, 1:-1] +
                     u_previous[2, 1:-1]) / dx2 + (u_previous[1, :-2] -
                                                   2 * u_previous[1, 1:-1] +
                                                   u_previous[1, 2:]) / dy2

    del_sqrd_u_down = (u_previous[-3, 1:-1] - 2 * u_previous[-2, 1:-1] +
                       u_previous[-1, 1:-1]) / dx2 + (
                           u_previous[-2, :-2] - 2 * u_previous[-2, 1:-1] +
                           u_previous[-2, 2:]) / dy2

    u[1, 1:-1] = u_previous[1, 1:-1] + dt * a * del_sqrd_u_up
    u[-2, 1:-1] = u_previous[-2, 1:-1] + dt * a * del_sqrd_u_down

    u_previous[:, :] = u[:, :]
