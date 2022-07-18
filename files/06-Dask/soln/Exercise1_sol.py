import dask
from dask.distributed import Client
import numpy as np
import time


def calculate_pi(size_in_bytes):
    
    """Calculate pi using a Monte Carlo method."""
    
    rand_array_shape = (int(size_in_bytes / 8 / 2), 2)
    
    # 2D random array with positions (x, y)
    xy = np.random.uniform(low=0.0, high=1.0, size=rand_array_shape)
    
    # check if position (x, y) is in unit circle
    xy_inside_circle = (xy ** 2).sum(axis=1) < 1

    # pi is the fraction of points in circle x 4
    pi = 4 * xy_inside_circle.sum() / xy_inside_circle.size

    print(f"\nfrom {xy.nbytes / 1e9} GB randomly chosen positions")
    print(f"   pi estimate: {pi}")
    print(f"   pi error: {abs(pi - np.pi)}\n")
    
    return pi


if __name__ == '__main__':

# Numpy only version
    t0 = time.time()
    for i in range(5):
       pi1 = calculate_pi(1000000000*(i+1))
    t1 = time.time()
    print(f"time taken for nupmy is {t1-t0}\n\n")

# Dask version
    client = Client(n_workers=5, threads_per_worker=1, memory_limit='100GB')
    client

    t0 = time.time()
    results = []
    for i in range(5):
       dask_calpi = dask.delayed(calculate_pi)(1000000000*(i+1))
       results.append(dask_calpi)
    dask.compute(*results)
    t1 = time.time()
    print("time taken for dask w5/t1 is " + str(t1-t0))


    client.close()
    

