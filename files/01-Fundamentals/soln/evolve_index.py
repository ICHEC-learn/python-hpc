import numpy as np

def evolve(u, u_previous, a, dt, dx2, dy2):

  

    del_sqrd_u = (u_previous[:-2, 1:-1] - 2*u_previous[1:-1, 1:-1] + u_previous[2:, 1:-1]) / dx2 + (u_previous[1:-1, :-2] - 2*u_previous[1:-1, 1:-1] + u_previous[1:-1, 2:]) / dy2 
    
    u[1:-1,1:-1] = u_previous[1:-1,1:-1] + dt * a *  del_sqrd_u

    u_previous[:,:] = u[:,:]