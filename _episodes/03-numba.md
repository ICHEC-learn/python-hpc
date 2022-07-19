---
title: "Numba"
teaching: 10
exercises: 0
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

And to top it off, you can run Numba code on GPU. This will not be covered in this lesson however.

When it comes to importing the correct materials, a common workflow (and one which will be replicated throughout the
episode) is as follows. It is very important to check the version of Numba that you have, as it is a rapidly evolving
library with many changes happening on a regular basis. In your own time, it is best to have an environment set up with
numba and update it regularly.

~~~
import numba
from numba import jit, njit

print(numba.__version__)
~~~
{: .language-python}

We will be covering the `jit` and `njit` operators in the upcoming section.

## The JIT Compiler Options/Toggles

Below we have a JIT decorator and as you can see there are plenty of different toggles and operations which we will
go through.

~~~
@numba.jit(signature=None, nopython=False, nogil=False, cache=False, forceobj=False, parallel=False, 
           error_model='python', fastmath=False, locals={}, boundscheck=False)
~~~
{: .language-python}

Looks like a lot, but let's go through the different options.

- `signature`: The expected types and signatures of function arguments and return values. This is known as an "Eager 
  Compilation".

- Modes: Numba has two modes; `nopython`, `forcobj`. Numba will infer the argument types at call time, and generate
  optimized code based on this information. If there is python object, the object mode will be used by default.

- `nogil=True`: Releases the global interperter lock inside the compiled function. This is one of the main reasons 
  python is considered as slow. However this only applies in `nopython` mode at present.

- `cache=True`: Enables a file-based cache to shorten compilation times when the function was already compiled in a
  previous invocation. It cannot be used in conjuction with `parallel=True`.

- `parallel=True`: Enables the automatic parallelization of a number of common Numpy constructs.

- `error_model`: This controls the divide-by-zero behavior. Setting it to `python` causes divide-by-zero to raise an
  exception. Setting it to `numpy` causes it to set the result to +/- infinity or `NaN`.

- `fastmath=True`: This enables the use of otherwise unsafe floating point transforms as described in the LLVM 
  documentation.

- `locals` dictionary: This is used to force the types and signatures of particular local variables. It is however 
  recommended to let Numba’s compiler infer the types of local variables by itself.

- `boundscheck=True`: This enables bounds checking for array indices. Out of bounds accesses will raise an `IndexError`. 
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
arguments presented, for example if we changed n = `1000.0`, we will get a longer execution time again, as the machine
code has had to be rewritten. 

To benchmark Numba-compiled functions, it is important to time them without including the compilation step. 
The compilation will only happen once for each set of input types, but the function will be called many times. By 
adding `@jit` decorator we see major speed ups for Python and a bit for NumPy. Numba is very useful in speeding up 
python loops that cannot be converted to NumPy or it's too complicated. NumPy can sometimes reduce readability. We can
therefore get huge speed ups with minimum effort.

## Demonstrating modes

{% include links.md %}
