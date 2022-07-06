def evolve(u, u_previous, a, dt, dx2, dy2):
    
    n = u.shape[0]
    m = u.shape[1]

    for i in range(1, n-1):
        for j in range(1, m-1):
            u[i, j] = u_previous[i, j] + a * dt * ( \
             (u_previous[i+1, j] - 2*u_previous[i, j] + \
              u_previous[i-1, j]) /dx2 + \
             (u_previous[i, j+1] - 2*u_previous[i, j] + \
                 u_previous[i, j-1]) /dy2 )
    u_previous[:] = u[:]