---
title: "Numba"
teaching: 50
exercises: 25
questions:
- "What is Just-In-Time compilation"
- "How can I implement Numba"
objectives:
- "Compare timings for python, NumPy and Numba"
- "Understand the different modes and toggles of Numba"
keypoints:
- "Numba only compiles individual functions rather than entire scripts."
- "The recommended modes are `nopython=True` and `njit`"
- "Numba is constantly changing, so keep checking for new versions."
---

<p align="center"><img src="../fig/ICHEC_Logo.jpg" width="40%"/></p>

## What is Numba?

The name [Numba](https://numba.pydata.org/) is the combination of the Mamba snake and NumPy, with Mamba being in 
reference to the black mamba, one of the world's fastest snakes. Numba is a just-in-time (JIT) compiler for Python 
functions. Numba can, from the types of the function arguments translate the function into a specialised, fast, machine
code equivalent. The basic principle is that you have a Python code written, then "wrap" it in a "jit" compiler.

Under the hood, it uses an LLVM compiler infrastructure for code generation. When utilised correctly, one can get
speeds similar to C/C++ and Fortran but without having to write any code in those languages, and no C/C++ compiler is 
required.

It sounds great, and thankfully, fairly simple. All one has to do is apply one of the Numba decorators to your Python 
function. A decorator is a function that takes in a function as an argument and spits out a function. As such, it is
designed to work well with NumPy arrays and is therefore very useful for scientific computing. As a result, it makes it
easy to parallelise your code and use multiple threads.

It works with SIMD vectorisation to get most out of your CPU. What this means is that a single instruction can be
applied to multiple data elements in parallel. Numba automatically translates some loops into vector instructions and
will adapt to your CPU capabilities automatically.

And to top it off, as it works with threading, you can run Numba code on GPU. This will not be covered in this lesson 
however, but we will cover GPUs in a later section.

When it comes to importing the correct materials, a common workflow (and one which will be replicated throughout the
episode) is as follows. 
~~~
import numba
from numba import jit, njit

print(numba.__version__)
~~~
{: .language-python}

It is very important to check the version of Numba that you have, as it is a rapidly evolving
library with many changes happening on a regular basis. In your own time, it is best to have an environment set up with
numba and update it regularly. We will be covering the `jit` and `njit` operators in the upcoming section.

## The JIT Compiler Options/Toggles

Below we have a JIT decorator and as you can see there are plenty of different toggles and operations which we will
go through.

~~~
@numba.jit(signature=None, nopython=False, nogil=False, cache=False, forceobj=False, parallel=False, 
           error_model='python', fastmath=False, locals={}, boundscheck=False)
~~~
{: .language-python}

Looks like a lot, but let's go through the different options.

- **`signature`**: The expected types and signatures of function arguments and return values. This is known as an "Eager 
  Compilation".

- **Modes**: Numba has two modes; `nopython`, `forcobj`. Numba will infer the argument types at call time, and generate
  optimized code based on this information. If there is python object, the object mode will be used by default.

- **`nogil=True`**: Releases the global interpreter lock inside the compiled function. This is one of the main reasons 
  python is considered as slow. However this only applies in `nopython` mode at present.

- **`cache=True`**: Enables a file-based cache to shorten compilation times when the function was already compiled in a
  previous invocation. It cannot be used in conjunction with `parallel=True`.

- **`parallel=True`**: Enables the automatic parallelization of a number of common Numpy constructs.

- **`error_model`**: This controls the divide-by-zero behavior. Setting it to `python` causes divide-by-zero to raise an
  exception. Setting it to `numpy` causes it to set the result to +/- infinity or `NaN`.

- **`fastmath=True`**: This enables the use of otherwise unsafe floating point transforms as described in the LLVM 
  documentation.

- **`locals` dictionary**: This is used to force the types and signatures of particular local variables. It is however 
  recommended to let Numba’s compiler infer the types of local variables by itself.

- **`boundscheck=True`**: This enables bounds checking for array indices. Out of bounds accesses will raise an `IndexError`. 
  Enabling bounds checking will slow down typical functions, so it is recommended to only use this flag for debugging.

## Comparing pure Python and NumPy, with and without decorators

Let us look at an example using the MonteCarlo method.

~~~
#@jit
def pi_montecarlo_python(n):
    in_circle = 0
    for i in range(int(n)):
        x, y = np.random.random(), np.random.random()
        if x ** 2 + y ** 2 <= 1.0:
            in_circle += 1
        
    return 4.0 * in_circle / n

#@jit
def pi_montecarlo_numpy(n):
    in_circle = 0
    x = np.random.random(int(n))
    y = np.random.random(int(n))
    in_circle = np.sum((x ** 2 + y ** 2) <= 1.0)
        
    return 4.0 * in_circle / n

n = 1000

print('python')
%time pi_montecarlo_python(n)
print("")
print('numpy')
%time pi_montecarlo_numpy(n)
~~~
{: .language-python}

~~~
python
CPU times: user 596 µs, sys: 1.69 ms, total: 2.29 ms
Wall time: 1.97 ms

numpy
CPU times: user 548 µs, sys: 0 ns, total: 548 µs
Wall time: 1.39 ms
~~~
{: .output}

> ## Adding the jit decorators
>
> Now, uncomment the lines with `#@jit`, and run it again. What execution times do you get, and do they improve with a 
> second run? Try reloading the function for a larger sample size. What difference do you get between normal code and
> code which has `@jit` decorators.
>
> > ## Output
> >
> > ~~~
> > python jit
> > CPU times: user 121 ms, sys: 9.4 ms, total: 131 ms
> > Wall time: 126 ms
> > 
> > numpy jit
> > CPU times: user 177 ms, sys: 1.85 ms, total: 179 ms
> > Wall time: 177 ms
> > 
> > SECOND RUN
> > 
> > python jit
> > CPU times: user 13 µs, sys: 5 ns, total: 18 µs
> > Wall time: 23.1 µs
> > 
> > numpy jit
> > CPU times: user 0 µs, sys: 21 µs, total: 21 µs
> > Wall time: 24.6 µs
> > ~~~
> > {: .output}
> >
> > You should have noticed that the wall time for the first run with the `@jit` decorators was significantly slower
> > than the original code. Then, when we run it again, it is much quicker So what's going on? 
> {: .solution}
{: .challenge}

The decorator is taking the python function and translating it into fast machine code, it will naturally take more time
to do so. This compilation time, is the time for numba to look through your code and translate it. If you are using a 
slow function and only using it once, then Numba will only slow it down. Therefore, Numba is best used for functions 
that you will be repeatedly using throughout your program.

Once the compilation has taken place Numba caches the machine code version of the function for the particular types of 
arguments presented, for example if we changed n to `1000.0` as a floating point number, we will get a longer execution
time again, as the machine code has had to be rewritten. 

To benchmark Numba-compiled functions, it is important to time them without including the compilation step. 
The compilation will only happen once for each set of input types, but the function will be called many times. By 
adding `@jit` decorator we see major speed ups for Python and a bit for NumPy. Numba is very useful in speeding up 
python loops that cannot be converted to NumPy or it's too complicated. NumPy can sometimes reduce readability. We can
therefore get significant speed ups with minimum effort.

## Demonstrating modes

### `nopython=True`

Below we have a small function that determines whether a function is a prime number, then generate an array of random
numbers. We are going to use a decorator for this example, which itself is a function that takes another function as
its argument, and returns another function, defined by `is_prime_jit`. This is as an alternative to using `@jit`.

~~~
def is_prime(n):
    if n <= 1:
        raise ArithmeticError('%s <= 1' %n)
    if n == 2 or n == 3:
        return True
    elif n % 2 == 0:
        return False
    else:
        n_sqrt = math.ceil(math.sqrt(n))
        for i in range(3, n_sqrt):
            if n % i == 0:
                return False

    return True

numbers = np.random.randint(2, 100000, size=10)

is_prime_jit = jit(is_prime)
~~~
{: .language-python}

Now we will time and run the function with pure python, jitted including compilation time and then purely with jit. 
Take note of the timing setup, as you will use this regularly through the episode.

~~~
print("Pure Python")
%time p1 = [is_prime(x) for x in numbers]
print("")
print("Jitted including compilation")
%time p2 = [is_prime_jit(x) for x in numbers]
print("")
print("Jitted")
%time p2 = [is_prime_jit(x) for x in numbers]
~~~
{: .language-python}

> ## Warning explanation
> 
> Upon running this, you will get an output with a large warning, amongst it will say,
> 
> ~~~
> Pure Python
> CPU times: user 798 µs, sys: 0 ns, total: 798 µs
> Wall time: 723 µs
> 
> Jitted including compilation
> ...
> ... Compilation is falling back to object with WITH looplifting enabled because Internal error in pre-inference
> rewriting pass encountered during compilation of function "is_prime" due to ... 
> ...
> 
> CPU times: user 482 ms, sys: 16.9 ms, total: 499 ms
> Wall time: 496 ms
> 
> Jitted
> CPU times: user 43 µs, sys: 0 ns, total: 43 µs
> Wall time: 46 µs
> ~~~
> {: .output}
> 
> This still runs as we would expect, and if we run it again, the warning disappears. 
{: .callout}

If we change the above code and add one of our toggles so that the jitted line becomes;

~~~
is_prime_jit = jit(nopython=True)(is_prime)
~~~
{: .language-python}

We will get a full error, because it **CANNOT** run in `nopython` mode. However, by setting this mode to `True`, you
can highlight where in the code you need to speed it up. So how can we fix it?

If we refer back to the code, and the error, it arises in the notation in Line 3 for the `ArithmeticError`, or more
specifically `'%s <= 1'`. This is a python notation, and to translate it into pure machine code, it needs the python
interpreter. We can change it to `'n <= 1'`, and when rerun we get no warnings or error.

Although what we have just done is possible without `nopython`, it is a bit slower, and worth bearing in mind.

`@jit(nopython=True)` is equivalent to `@njit`. The behaviour of the `nopython` compilation mode is to essentially 
compile the decorated function so that it will run entirely without the involvement of the Python interpreter. If it
can't do that an exception is raised. These exceptions usually indicate places in the function that need to be modified
in order to achieve better-than-Python performance. Therefore, we strongly recommend always using `nopython=True`. This
supports a subset of python but runs at C/C++/Fortran speeds.

Object mode (`forceobj=True`) extracts loops and compiles them in `nopython` mode which useful for functions that are 
bookended by uncompilable code but have a compilable core loop, this is also done automatically. It supports nearly all
of python but cannot speed up by a large factor.

## Mandelbrot example

Let's now create an example of the Mandelbrot set, or strictly speaking, a Julia set in reality. We won't go into
full details on what is going on in the code, but there is a `while` loop in the `kernel` function that is causing this
to be slow as well as a couple of `for` loops in the `compute_mandel_py` function.

~~~
import numpy as np
import math
import time
import numba
from numba import jit, njit
import matplotlib.pyplot as plt

mandel_timings = []

def plot_mandel(mandel):
    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.imshow(mandel, cmap='gnuplot')
    plt.savefig('mandel.png')
    
def kernel(zr, zi, cr, ci, radius, num_iters):
    count = 0
    while ((zr*zr + zi*zi) < (radius*radius)) and count < num_iters:
        zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
        count += 1
    return count

def compute_mandel_py(cr, ci, N, bound, radius=1000.):
    t0 = time.time()
    mandel = np.empty((N, N), dtype=int)
    grid_x = np.linspace(-bound, bound, N)

    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_x):
            mandel[i,j] = kernel(x, y, cr, ci, radius, N)
    return mandel, time.time() - t0

def python_run():
    kwargs = dict(cr=0.3852, ci=-0.2026,
              N=400,
              bound=1.2)
    print("Using pure Python")
    mandel_func = compute_mandel_py       
    mandel_set = mandel_set, runtime = mandel_func(**kwargs)
    
    print("Mandelbrot set generated in {} seconds".format(runtime))
    plot_mandel(mandel_set)
    mandel_timings.append(runtime)

python_run()

print(mandel_timings)
~~~
{: .language-python}

For larger values of `N` in `python_run`, we recommend submitting this to the compute nodes. For more information on
submitting jobs on an HPC, you can consult our [intro-to-hpc](https://ichec-learn.github.io/intro-to-hpc/) course.

For the moment however we can provide you with the job script in an exercise.

> ## Submit a job to the queue
>
> Below is a job script that we have prepared for you. All you need to do is run it! This script will run the code
> above, which can be written to a file called `mandel.py`. Your instructor will inform you of the variables to use
> in the values `$ACCOUNT`, `$PARTITION`, and `$RESERVATION`. 
> 
> ~~~
> #!/bin/bash
> #SBATCH --nodes=1
> #SBATCH --time=00:10:00
> #SBATCH -A $ACCOUNT
> #SBATCH --job-name=mandel
> #SBATCH -p $PARTITION
> #SBATCH --reservation=$RESERVATION
> 
> module purge
> module load conda
> module list
> 
> source activate numba
> 
> cd $SLURM_SUBMIT_DIR
> 
> python mandel.py
> 
> exit 0
> ~~~
> {: .language-bash}
>
> To submit the job, you will need the following command. It will return a job ID.
> 
> ~~~
> $ sbatch mandel_job.sh
> ~~~
> {: .language-bash}
> 
> Once the job has run successfully, run it again but this time, use `njit` on the `kernel` function.
>
> > ## Output
> >
> > Check the directory where you submitted the command and you will have a file called `slurm-123456.out`, where the
> > `123456` will be replaced with your job ID as returned in the previous example. View the contents of the file and
> > it will give you an output returning the time taken for the mandelbrot set. For `N = 400` it will be roughly 10-15
> > seconds. A `.png` file will also be generated.
> > 
> > To view the `njit` solution, head to the file [here](../files/02-Numba/exercise/02-Numba-Exercises-Solutions.ipynb).
> {: .solution}
{: .challenge}

Now, let's see if we can speed it up more by looking at the compute function.

~~~
compute_mandel_njit_jit = njit()(compute_mandel_njit)

def njit_njit_run():
    kwargs = dict(cr=0.3852, ci=-0.2026,
              N=400,
              bound=1.2)
    print("Using njit kernel and njit compute")
    mandel_func = compute_mandel_njit_jit      
    mandel_set = mandel_set, runtime = mandel_func(**kwargs)
    
    print("Mandelbrot set generated in {} seconds".format(runtime))

njit_njit_run()
~~~
{: .language-python}

> ## `njit` the compute function
>
> Add the modifications to your code and submit the job. What kind of speed up do you get?
>
> > ## Solution
> >
> > Not even a speed-up, an error, because there are items in that function that Numba does not like! More specifically
> > if we look at the error;
> > 
> > ~~~
> > ...
> > Unknown attribute 'time' of type Module ...
> > ...
> > ~~~
> > {: .output}
> >
> > This is a python object, so we need to remove the `time.time` function.
> {: .solution}
{: .challenge}

Now let's modify the code to ensure it is timed outside of the main functions. Running it again will produce another
error about data types, so these will also need to be fixed.

> ## Fix the errors!
> 
> Change all the instances of `dtype=int` to `dtype=np.int_` in `compute_mandel` functions throughout.
> 
> > ## Solution
> >
> > ~~~
> > import matplotlib.pyplot as plt
> > 
> > mandel_timings = []
> > 
> > def plot_mandel(mandel):
> >     fig=plt.figure(figsize=(10,10))
> >     ax = fig.add_subplot(111)
> >     ax.set_aspect('equal')
> >     ax.axis('off')
> >     ax.imshow(mandel, cmap='gnuplot')
> > 
> > def kernel(zr, zi, cr, ci, radius, num_iters):
> >     count = 0
> >     while ((zr*zr + zi*zi) < (radius*radius)) and count < num_iters:
> >         zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
> >         count += 1
> >     return count
> > 
> > kernel_njit = njit(kernel)
> > 
> > def compute_mandel_njit(cr, ci, N, bound, radius=1000.):
> >     mandel = np.empty((N, N), dtype=np.int_)
> >     grid_x = np.linspace(-bound, bound, N)
> > 
> >     for i, x in enumerate(grid_x):
> >         for j, y in enumerate(grid_x):
> >             mandel[i,j] = kernel_njit(x, y, cr, ci, radius, N)
> >     return mandel
> > 
> > compute_mandel_njit_jit = njit()(compute_mandel_njit)
> > 
> > def njit_njit_run():
> >     kwargs = dict(cr=0.3852, ci=-0.2026,
> >               N=200,
> >               bound=1.2)
> >     print("Using njit kernel and njit compute")
> >     mandel_func = compute_mandel_njit_jit      
> >     mandel_set = mandel_func(**kwargs)
> >    
> >     plot_mandel(mandel_set)
> > 
> > njit_njit_run()
> > ~~~
> > {: .language-python}
> > 
> > We recommend trying this out in the [Jupyter notebook](../files/02-Numba/02-Numba.ipynb) as well for your own reference
> > 
> > ~~~
> > t0 = time.time()
> > njit_njit_run()
> > runtime = time.time() - t0
> > mandel_timings.append(runtime)
> > print(mandel_timings)
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

### `cache=True`

The point of using `cache=True` is to avoid repeating the compile time of large and complex functions at each run of a
script. In the example below the function is simple and the time saving is limited but for a script with a number of
more complex functions, using cache can significantly reduce the run-time. We have removed the python object that 
caused the error. We will switch back to the `is_prime` function here.

~~~
def is_prime(n):
    if n <= 1:
        raise ArithmeticError('n <= 1')
    if n == 2 or n == 3:
        return True
    elif n % 2 == 0:
        return False
    else:
        n_sqrt = math.ceil(math.sqrt(n))
        for i in range(3, n_sqrt):
            if n % i == 0:
                return False

    return True

is_prime_njit = njit()(is_prime)
is_prime_njit_cached = njit(cache=True)(is_prime)

numbers = np.random.randint(2, 100000, size=1000)
~~~
{: .language-python}

> ## Compare the timings for `cache`
>
> Run the function 4 times so that you get results for:
> - Not cached including compilation
> - Not cached 
> - Cached including compilation
> - Cached
>
> > ## Output
> > 
> > You may get a result similar to below.
> > ~~~
> > Not cached including compilation
> > CPU times: user 117 ms, sys: 11.7 ms, total: 128 ms
> > Wall time: 134 ms
> > 
> > Not cached
> > CPU times: user 0 ns, sys: 381 µs, total: 381 µs
> > Wall time: 386 µs
> > 
> > Cached including compilation
> > CPU times: user 2.84 ms, sys: 1.95 ms, total: 4.79 ms
> > Wall time: 11.8 ms
> > 
> > Cached
> > CPU times: user 378 µs, sys: 0 ns, total: 378 µs
> > Wall time: 382 µs
> > ~~~
> > {: .output}
> >
> > It may not be as fast as when its been compiled in the same environment you are running your program in, but can
> > still be a considerable speed up for bigger scripts. Usually to show the cache working, you need to restart the 
> > whole kernel and subsequently reload the modules, functions and variables.
> >
> {: .solution}
{: .challenge}

### Eager Compilation using function signatures

This speeds up compilation time faster than cache, hence the term "eager". It can be helpful if you know the types of
input and output values of your function before you compile it. Although python can be fairly lenient if you are not 
concerned about types, at the machine level it makes a big difference. We will look more into the importance of typing
in upcoming episodes, but for now, let's look again at our prime example. We do not need to edit the code itself, 
merely the njit.

To enable eager compilation, we need to specify the input and output types. For `is_prime`, the output is a boolean,
and the input is an integer, we have to specify that as well. It needs to be declared in the form `output(input)`.
Types should be ordered from smaller to higher precision, i.e. `int32`, `int64`. We have to cover for both methods of
precision.

~~~
is_prime_eager = njit(['boolean(int32)','boolean(int64)' ])(is_prime)
~~~
{: .language-python}

> ## Compare the timings for eager compilation
>
> Run the function 4 times so that you get results for:
> - Just njit including compilation
> - Just njit
> - Eager including compilation
> - Eager
>
> > ## Output
> > 
> > You may expect an output similar to the following.
> > 
> > ~~~
> > Just njit including compilation
> > CPU times: user 97.2 ms, sys: 2.19 ms, total: 99.4 ms
> > Wall time: 100 ms
> > 
> > Just njit
> > CPU times: user 3.6 ms, sys: 0 ns, total: 3.6 ms
> > Wall time: 3.48 ms
> > 
> > Eager including compilation
> > CPU times: user 3.61 ms, sys: 0 ns, total: 3.61 ms
> > Wall time: 3.57 ms
> > 
> > Eager
> > CPU times: user 3.42 ms, sys: 367 µs, total: 3.79 ms
> > Wall time: 3.64 ms
> > ~~~
> > {: .output}
> >
> > Bear in mind these are small examples, but you can clearly see how much time that has shaved off this small 
> > example .
> >
> {: .solution}
{: .challenge}

### `fastmath=True`

The final one we will look at for `is_prime` is `fastmath=True`. This enables the use of otherwise unsafe floating
point transforms. This means that it is possible to relax some numerical rigour with view of gaining additional 
performance. As an example, it assumes that the arguments and result are not `NaN` or infinity values. Feel free to 
investigate the [llvm docs](https://llvm.org/docs/LangRef.html#fast-math-flags). The key thing with this toggle is that
you have to be confident with the inputs of your code and that there is no chance of returning `NaN` or infinity.

~~~
is_prime_njit_fmath = njit(fastmath=True)(is_prime)
~~~
{: .language-python}

Running this, you may expect an timings output similar to below.

~~~
CPU times: user 3.75 ms, sys: 0 ns, total: 3.75 ms
Wall time: 3.75 ms

Fastmath including compilation
CPU times: user 96 ms, sys: 477 µs, total: 96.5 ms
Wall time: 93.9 ms

Fastmath compilation
CPU times: user 3.5 ms, sys: 0 ns, total: 3.5 ms
Wall time: 3.41 ms
~~~
{: .output}

> ## Toggling with toggles with Montecarlo
>
> Head to the Numba Exercise [Jupyter notebook](../files/02-Numba/exercise/02-Numba-Exercises.ipynb) and work on
> Exercise 2. You should try out `@njit`, eager compilation, `cache` and `fastmath` on the MonteCarlo function 
> and compare the timings you get. Feel free to submit larger jobs to the queue.
> 
> > ## Solution
> > 
> > The solution can be found in the notebook [here](../files/02-Numba/exercise/02-Numba-Exercises-Solutions.ipynb).
> >
> {: .solution}
{: .challenge}

### `parallel=True`

We can also use Numba to parallelise our code by using `parallel=True` to use multi-core CPUs via threading. We can use
`numba.prange` alongside `parallel=True` if you have for loops present in your code. As a default, the option is set to
`False`, and doing so means that `numba.prange` has the same utility as `range`. We can set the default number of 
threads with the following syntax. 

~~~
max_threads = numba.config.NUMBA_NUM_THREADS
~~~
{: .language-python}

~~~
def pi_montecarlo_python(n):
    in_circle = 0
    for i in range(n):
        x, y = np.random.random(), np.random.random()
        if x ** 2 + y ** 2 <= 1.0:
            in_circle += 1
        
    return 4.0 * in_circle / n

def pi_montecarlo_numpy(n):
    in_circle = 0
    x = np.random.random(n)
    y = np.random.random(n)
    in_circle = np.sum((x ** 2 + y ** 2) <= 1.0)
        
    return 4.0 * in_circle / n

n = 1000000

pi_montecarlo_python_njit = njit()(pi_montecarlo_python)

pi_montecarlo_numpy_njit = njit()(pi_montecarlo_numpy)

pi_montecarlo_python_parallel = njit(parallel=True)(pi_montecarlo_python)

pi_montecarlo_numpy_parallel = njit(parallel=True)(pi_montecarlo_numpy)
~~~
{: .language-python}

If the pure python version seems faster than numpy, there is no need for concern, as sometimes python + numba can turn
out to be faster than numpy + numba. 

> ## Explaining warnings
>
> If you run the above code, you may see that you get a warning saying:
>
> ~~~
> ...
> The keyword argument 'parallel=True' was specified but no transformation for parallel executing code was possible.
> ...
> ~~~
> {: .output}
> 
> Running it again will remove the warning, but we will not get any speed-up. We will need to change the above code in
> Line 3 from `for i in range(n):` to `for i in numba.prange(n):`.
{: .callout}

~~~
njit_python including compilation
CPU times: user 105 ms, sys: 4.66 ms, total: 110 ms
Wall time: 105 ms

njit_python
CPU times: user 10.1 ms, sys: 0 ns, total: 10.1 ms
Wall time: 9.93 ms

njit_numpy including compilation
CPU times: user 174 ms, sys: 7.61 ms, total: 181 ms
Wall time: 179 ms

njit_numpy
CPU times: user 11.1 ms, sys: 4.3 ms, total: 15.4 ms
Wall time: 15.2 ms

njit_python_parallel including compilation
CPU times: user 536 ms, sys: 29.1 ms, total: 565 ms
Wall time: 480 ms

njit_python_parallel
CPU times: user 60.3 ms, sys: 8.65 ms, total: 68.9 ms
Wall time: 3.2 ms

njit_numpy_parallel including compilation
CPU times: user 3.89 s, sys: 726 ms, total: 4.62 s
Wall time: 789 s

njit_numpy_parallel
CPU times: user 53.1 ms, sys: 9.96 ms, total: 63 ms
Wall time: 2.77 ms
~~~
{: .output}

> ## Set the number of threads
>
> Increase the value of N and adjust the number of threads using `numba.set_num_threads(nthreads)`. What sort of
> timings do you get. Are they what you would expect? Why?
> 
{: .challenge}

> ## Diagnostics
>
> Using the command below:
>
> ~~~
> pi_montecarlo_numpy_parallel.parallel_diagnostics(level=N)
> ~~~
> {: .language-python}
>
> You can get an understanding of what is going on under the hood. You can replace the value N for numbers between 1 
> and 4. 
>
{: .callout}

### Creating `ufuncs` using `numba.vectorize`

A universal function (or `ufunc` for short) is a function that operates on ndarrays in an element-by-element fashion.
So far we have been looking at just-in-time wrappers, these are “vectorized” wrappers for a function. For example 
`np.add()` is a ufunc.

There are two main types of ufuncs:
- Those which operate on scalars, ufuncs (see `@vectorize` below).
- Those which operate on higher dimensional arrays and scalars, these are “generalized universal functions” or gufuncs, 
  such as `@guvectorize`.

The` @vectorize` decorator allows Python functions taking scalar input arguments to be used as NumPy `ufuncs`. Creating
a traditional NumPy ufunc involves writing some C code. Thankfully, Numba makes this easy. This decorator means Numba
can compile a pure Python function into a `ufunc` that operates over NumPy arrays as fast as traditional ufuncs written
in C. 

The `vectorize()` decorator has two modes of operation:

1. **Eager**, or decoration-time, compilation: If you pass one or more type signatures to the decorator, you will be 
  building a Numpy universal function (ufunc). It is passed in the formw

~~~
output_type1(input_type1)
output_type2(input_type12)
# etc
~~~
{: .language-python}

2. **Lazy**, or call-time, compilation: When not given any signatures, the decorator will give you a Numba dynamic
  universal function (DUFunc) that dynamically compiles a new kernel when called with a previously unsupported input 
  type.

If you pass several signatures, beware that you have to pass the more specific signatures before least specific ones
(e.g., single-precision floats before double-precision floats), otherwise type-based dispatching will not work as
expected. eg (`int32`,`int64`,`float32`,`float64`)

Here is a very simple example one with vectorization and the other with parallelisation as well.

~~~
numba.set_num_threads(max_threads)

def numpy_sin(a, b):
    return np.sin(a) + np.sin(b) + np.cos(a) - np.cos(b) + (np.sin(a))**2


numpy_sin_vec = numba.vectorize(['int64(int64, int64)','float64(float64, float64)'])(numpy_sin)

numpy_sin_vec_par = numba.vectorize(['int64(int64, int64)','float64(float64, float64)'],target='parallel')(numpy_sin)

x = np.random.randint(0, 100, size=90000)
y = np.random.randint(0, 100, size=90000)

print("Just numpy")
%time _ = numpy_sin(x, y)
print("")
print("Vectorised")
%time _ = numpy_sin_vec(x, y)
print("")
print("Vectorised & parallelised")
%time _ = numpy_sin_vec_par(x, y)
~~~
{: .language-python}

~~~
Just numpy
CPU times: user 17.3 ms, sys: 4.08 ms, total: 21.4 ms
Wall time: 20.1 ms

Vectorised
CPU times: user 14.9 ms, sys: 0 ns, total: 14.9 ms
Wall time: 14.5 ms

Vectorised & parallelised
CPU times: user 86.5 ms, sys: 18.7 ms, total: 105 ms
Wall time: 8.72 ms
~~~
{: .output}

> ## Vectorisation
> 
> Head to the [Jupyter notebook](../files/02-Numba/exercise/02-Numba-Exercises.ipynb) and Exercise 3. Work on the 
> `is_prime` function by utilising `njit`, `vectorize` and then vectorize it with the target set to `parallel`. Time
> the results and compare the output.
> 
> > ## Solution
> >
> > The solution can be found in the notebook [here](../files/02-Numba/exercise/02-Numba-Exercises-Solutions.ipynb).
> > 
> {: .solution}
{: .challenge}

{% include links.md %}