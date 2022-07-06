import matplotlib.pyplot as plt
import numpy as np
import time
from evolve_index import evolve

# Set the colormap
plt.rcParams['image.cmap'] = 'BrBG'

# Basic parameters
a = 0.5                # Diffusion constant
timesteps = 200        # Number of time-steps to evolve system
image_interval = 4000  # Write frequency for png files

# Grid spacings
dx = 0.01
dy = 0.01
dx2 = dx**2
dy2 = dy**2

# For stability, this is the largest possible size of the time-step:
dt = dx2*dy2 / ( 2*a*(dx2+dy2) )

def initial_fields(filename):
    field = np.loadtxt(filename)
    field0 = np.copy(field)
    return field, field0

def write_field(field, step):
    plt.gca().clear()
    plt.imshow(field)
    plt.axis('off')
    plt.savefig('heat_{0:03d}.png'.format(step))

def iterate(field, field0, timesteps, image_interval):
    for i in range(1,timesteps+1):
        evolve(field, field0, a, dt, dx2, dy2)
        if i % image_interval == 0: # never goes in..?
            write_field(field, i)

def main():
    field, field0 = initial_fields('bottle.dat')
    write_field(field, 0)

    t0 = time.time()
    iterate(field, field0, timesteps, image_interval)
    t1 = time.time()

    write_field(field, timesteps)
    print ("Running Time: {0}".format(t1-t0))

main()




