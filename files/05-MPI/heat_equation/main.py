
import matplotlib.pyplot
import time
import numpy
from mpi4py import MPI
from mpi4py.MPI import Request
from modules import evolve, io, comms

# Set the colormap
matplotlib.pyplot.rcParams['image.cmap'] = 'BrBG'

# Basic parameters
a = 0.5  # Diffusion constant
timesteps = 1000  # Number of time-steps to evolve system
image_interval = 4000  # Write frequency for png files

# Grid spacings
dx = 0.01
dy = 0.01
dx2 = dx**2
dy2 = dy**2

# For stability, this is the largest possible size of the time-step:
dt = dx2 * dy2 / (2 * a * (dx2 + dy2))


def iterate(field, field0, timesteps, image_interval):
    for i in range(1, timesteps + 1):
        evolve.evolve(field, field0, a, dt, dx2, dy2)


def main():
    print('Initialising field...')
    field = io.initialise_field('data/bottle.dat')
    
    io.write_field(field, 0)
    print('Initial field written to output/bottle_0000.png.')

    field0 = field[:, :]

    print('Simulating...')
    t0 = time.time()
    iterate(field, field0, timesteps, image_interval)
    t1 = time.time()
    print('Simulation complete.')
    io.write_field(field, timesteps)
    print('Final field written to output/bottle_{}.png.'.format(timesteps))
    print("Running Time: {0}".format(t1 - t0))


main()
