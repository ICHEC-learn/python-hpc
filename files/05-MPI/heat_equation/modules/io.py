
import numpy
import matplotlib.pyplot


def initialise_field(filename):
    field = numpy.loadtxt(filename)
    return field


def write_field(field, step):
    matplotlib.pyplot.gca().clear()
    matplotlib.pyplot.imshow(field)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.savefig('output/bottle_{0:04d}.png'.format(step))


def to_text_file(field, filename):
    numpy.savetxt(filename, field, fmt='%1.1f')
