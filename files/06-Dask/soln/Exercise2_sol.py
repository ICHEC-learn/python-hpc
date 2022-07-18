import numpy as np

import dask
import dask.array as da
from dask.distributed import Client


import time


def dask_calculate_pi(size_in_bytes,nchunks):
    
    """Calculate pi using a Monte Carlo method."""
    
    rand_array_shape = (int(size_in_bytes / 8 / 2), 2)
    chunk_size = int(rand_array_shape[0]/nchunks)
    print(chunk_size)
    
    # 2D random array with positions (x, y)
    xy = da.random.uniform(low=0.0, high=1.0, size=rand_array_shape, chunks=chunk_size)
    print(f" Created xy\n {xy}\n")
    print(f" Number of partitions/chunks is {xy.numblocks}\n")
    
    
    # check if position (x, y) is in unit circle
    xy_inside_circle = (xy ** 2).sum(axis=1) < 1

    # pi is the fraction of points in circle x 4
    pi = 4 * xy_inside_circle.sum() / xy_inside_circle.size
    
    result = pi.compute()

    print(f"\nfrom {xy.nbytes / 1e9} GB randomly chosen positions")
    print(f"   pi estimate: {result}")
    print(f"   pi error: {abs(result - np.pi)}\n")
    
    return result


if __name__ == '__main__':


    client = Client(n_workers=5, threads_per_worker=4, memory_limit='40GB')
    print(client)

    t0 = time.time()
    dask_calculate_pi(100000000000,40)
    t1 = time.time()
    print("time taken for dask is " + str(t1-t0))

    client.close()


