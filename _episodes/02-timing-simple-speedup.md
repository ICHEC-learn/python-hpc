---
title: "Timing Code and Simple Speed-up Techniques"
teaching: 10
exercises: 0
questions:
- "How can I time my code"
- "What is the difference between vectorisation and for loops?"
- "What is the cache?"
- "How can `lru_cache` speed up my code?"
objectives:
- "Introduce `time`, `timeit` modules for timing code and `cProfile`, `pstats` for code profiling"
keypoints:
- "Performance code profiling is used to identify and analyse the execution and improvement of applications."
- "Never try and optimise your code on the first try. Get the code correct first."
- "Most often, about 90% of the code time is spent in 10% of the application."
- "The `lru_cache()` helps reduce the execution time of the function by using the memoization technique, discarding
   least recently used items first."
---

<p align="center"><img src="../fig/ICHEC_Logo.jpg" width="40%"/></p>

## Performance Measurement

Before diving into methods to speed up a code, we need to look at some tools which can help us understand how long our
code actually takes to run. This is the most fundamental tool and one of the most important processes that one has to
go through as a programmer.

Performance code profiling is a tool used to identify and analyse the execution of applications, and identify the 
segments of code that can be improved to achieve a better speed of the calculations or a better flow of information and
memory usage.

This form of dynamic program analysis can provide information on several aspects of program optimization, such as:
- how long a method/routine takes to execute
- how often a routine is called
- how memory allocations and garbage collections are tracked
- how often web services are called

It is important to never try and optimise your code on the first try. Get the code correct first. It is often said that
premature optimization is the root of all evil when it comes to programming.

Before optimising a code, one needs to find out where the time is spent most. Around 90% of the time is spent in 10% of
the application.

There are a number of different methods and modules one can use
- `time`
- `timeit`
- `cProfile`
- `datetime`
- `astropy` - mainly for astronomy usage
- Full fledged profiling tools: TAU, Intel Vtune, Python Tools for Visual Studio, etc.

### `time` module

Python's time module can be used for measuring time spent in specific part of the program. It can give the absolute 
real-world time measured from a fixed point in the past using `time.time()`. Additionally, `time.perf_counter()` and
`time.process_time()` can be used to derive the relative time. unit-less value which is proportional to the time 
elapsed between two instants.

~~~
import time

print (time.time())

print (time.perf_counter()) # includes time elapsed during sleep (CPU counter)
print (time.process_time()) # does not include time elapsed during sleep
~~~
{: .language-python}

~~~
1652264171.399796
2.433484803
0.885057
~~~
{: .output}

It's main use though is to take the `time.time()` function and assign it to a variable, then have the function which
you want to time, followed by a second call of the `time.time()` function and assign it to the new variable. A quick
arithmetic operation will quickly deduce the length of time taken for a specific function to run.

~~~
t0 = time.time()

my_list = []
for i in range(500): 
    my_list.append(0)
    
t1 = time.time()

tf = t1-t0

print('Time taken in for loop: ', tf)
~~~
{: .language-python}

~~~
Time taken in for loop:  0.00011992454528808594
~~~
{: .output}

### `timeit` module

This can be particularly useful as it can work in both python files and most importantly in the command line interface.
Although it can be used in both, it's use is excellent in the command line. The `timeit` module provides easy timing
for small bits of Python code, whilst also avoiding the common pitfalls in measuring execution times. The syntax from
the command line is as follows:

~~~
$ python -m timeit -s "from my module import func" "func()"
~~~
{: .language-bash}

~~~
10 loops, best of 3: 433 msec per loop
~~~
{: .output}

In a Python interface such as iPython, one can use magics (`%`).

~~~
In [1]: from mymodule import func
In [2]: %timeit func()
~~~
{: .language-bash}

~~~
10 loops, best of 3: 433 msec per loop
~~~
{: .output}

Let us look at an example using the Montecarlo technique to estimate the value of pi.

~~~
import random as rnd
import math


def montecarlo_py(n):
    x, y = [], []
    for i in range(n):
        x.append(rnd.random())
        y.append(rnd.random())
    pi_xy = [(x[i], y[i]) for i in range(n) if math.sqrt(x[i] ** 2 + y[i] ** 2) <= 1]
    return(4 * len(pi_xy) / len(x))
    # Estimate for pi
~~~
{: .language-python}

Our modules `math`, `random` are imported for the calculation, and the function returns an estimate for pi given `n`
number of points. Running the program can be done as so, and will produce a result close to `3.14`.

~~~
montecarlo_py(1000000)
~~~
{: .language-python}

If we want to time this using `timeit` we can modify the above statement using cell magics, and it will not produce the
result, but rather the average duration

~~~
%timeit montecarlo_py(1000000)
~~~
{: .language-python}

~~~
724 ms ± 6.52 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
~~~
{: .output}

You can implement a more controlled setup, where one sets the number of iterations, repetitions, etc, subject to the
use case.

~~~
import timeit

setup_code = "from __main__ import montecarlo_py"
stmt = "montecarlo_py(1000000)"
times = timeit.repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)
print(f"Minimum execution time: {min(times)}")
~~~
{: .language-python}

~~~
Minimum execution time: 7.153229266
~~~
{: .output}

### `cProfile`



{% include links.md %}
