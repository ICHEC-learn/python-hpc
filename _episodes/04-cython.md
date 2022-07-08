---
title: "Cython"
teaching: 10
exercises: 0
questions:
- "What is Cython?"
- "What's happening under the hood?"
- "How can I implement a Cython workflow?"
- "How can I create C-type declarations?"
objectives:
- "To understand the main concepts of C relative to Cython"
- "Learn how to implement a Cython workflow in a Jupyter notebook and in a terminal environment]9oii "
- "Implement type declarations across variables and functions, as well as compiler directives"
- "Undertake a Julia-set example"
keypoints:
- "Cython IS Python, only with C datatypes"
- "Working from the terminal, a `.pyx`, `main.py` and `setup.py` file are required."
- "From the terminal the code can be run with `python setup.py build_ext --inplace`"
- "Cython 'typeness' can be done using the `%%cython -a` cell magic, where the yellow tint is coloured according to
  typeness."
---

<p align="center"><img src="../fig/ICHEC_Logo.jpg" width="40%"/></p>

## What is Cython?

Cython is a programming language that makes writing C extensions for the Python language as easy as Python itself. The
source code gets translated into optimised C/C++ code and compiled as Python extension modules. The code is executed in
the CPython runtime environment, but at the speed of compiled C with the ability to call directly into C libraries, 
whilst keeping the original interface of the Python source code.

This enables Cython's two major use cases:

1. Extending the CPython interpreter with fast binary modules
2. Interfacing Python code with external C libraries

An important thing to remember is that Cython **IS** Python, only with C data types, so lets take a little but of time
to get into some datatypes.

## Typing

Cython supports static type declarations, thereby turning readable Python code into plain C performance. There are two
main recognised ways of "typing".

**Static Typing**

Type checking is performed during compile-time. For example, the expression `x = 4 + 'e'` would not compile. This
method of typing can detect type errors in rarely used code paths

**Dynamic Typing**

In contrast, type checking is performed during run-time. Here, the expression `x = 4 + 'e'` would result in a runtime
type error. This allows for fast program execution and tight integration with external C libraries.

> ## Datatype declarations
> Python is a programming language with an exception to the normal rule of datatype declaration. Across most 
> programming languages you will see variables associated with a specific type, such as integers (`int`), floats
> (`float`, `double`), and strings (`str`).
>
> We see datatypes used in pure Python when declaring things as a `list` or `dict`. For most Cython operations we will
> be doing the same for all variable types.
>
{: .callout}

## Implementing Cython

This can be implemetned either by Cython scripts or by using cell magics (`%%`) in Jupyter notebooks. Although the
notebooks are available, we also recommend trying these methods outside the notebook, as it is the more commonly used 
implementation. You can use Jupyter notebooks to create external files as an alternative.

> ## Using Jupyter Notebook cell magics with Cython
> 
> As discussed in [episode 2](02-timing-simple-speedup.md), we can use cell magics to implement Cython in Jupyter
> notebooks. The first cell magic that we need is to load Cython itself. This can be done with a separate cell block
> that only needs to be run once in the notebook.
>
> ~~~
> %load_ext cython
> ~~~
> {: .language-python}
> 
> From there, any cell that we wish to "cythonise" needs to have the following on the first line of any cell.
>
> ~~~
> %%cython
>
> # Some stuff
> ~~~
> {: .language-python}
{: .callout}

Let's look at how to cythonise a simple function which calculates the Fibonacci sequence for a given set of numbers, 
`n`. Below we have the python code for a file which we have named `fibonacci.py`. 

~~~
def fib(n):
    # Prints the Fibonacci series up to n.
    a, b = 0, 1
    while b < n:
        print(b)
        a, b = b, a + b
~~~
{: .language-python}

Although we are only dealing with one file here, it is common practice to have a "main" file from which all other files
and functions are called. This may seem unnecessary for this simple setup, but is good practice, particularly when you
have a setup that has dozens, hundreds of functions. We will create a simple `fibonacci_main.py` file which imports the
`fib` function, then runs it with a fixed value for `n`.

~~~
from fibonacci import fib

fib(10)
~~~
{: .language-python}

From here we can run the file in the terminal, or if you are using Jupyter notebook, you can use the cells themselves,
or use the `!` operator to implement bash in the codeblock.

~~~
$ python fibonacci_main.py
~~~
{: .language-bash}

That's our Python setup, now let's go about cythonising it. We will use the same function as before, but now we will
save it as a `.pyx` file. It can be helpful when dealing with Cython to rename your functions accordingly, as we can
see below.

~~~
def fib_cyt(n):
    # Prints the Fibonacci series up to n.
    a, b = 0, 1
    while b < n:
        print(b)
        a, b = b, a + b
~~~
{: .language-python}

Before we change our `fibonacci_main.py` to implement the function using Cython, we need to ado a few more things.
This `.pyx` file is compiled by Cython into a `.c` file, which itself is then compiled by a C compiler to a `.so` or
`.dylib` file. We will learn a bit more about these different file types in the [next episode](05.cffi.md).

There are a few different ways to build your extension module. We are going to look at a method which creates a file
which we will call `setup_fib.py`, which "cythonises" the file. It can be viewed as the `Makefile` of python. In it, we
need to import some modules and call a function which enables us to setup the file.

~~~
from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize("fibonacci_cyt.pyx"))
~~~
{: .language-python}

Let us have a look at the contents of our current working directory, and see how it changes as we run this setup file.

~~~
$ ls
~~~
{: .language-bash}

~~~
04-Cython.ipynb   exercise          fibonacci_cyt.pyx setup_fib.py
__pycache__       fibonacci.py      fibonacci_main.py     
~~~
{: .output}

At this stage all we have are our original Python files, our .pyx file and setup_fib.py. Now lets run our 
`setup_fib.py` and see how that changes.

In the terminal (or by using `!` in notebook), we will now build the extension to use in the current working directory.

~~~

~~~
{: .language-bash}

You should get a printed message to the screen, now let's check the output of `ls` to see how our directory has changed.

~~~
04-Cython.ipynb                    fibonacci_cyt.c
__pycache__                        fibonacci_cyt.cpython-38-darwin.so
build                              fibonacci_cyt.pyx
exercise                           fibonacci_main.py
fibonacci.py                       setup_fib.py
~~~
{: .output}

So, a few more things have been added now.

1. The `.c` file, which is then compiled using a C compiler
2. The `build/` directory, which contains the `.o` file generated by the compiler
3. The `.so` or `.dylib` file. This is the compiled library

Next we add the `main` file which we will use to run our program. We can call it `fibonacci_cyt_main.py`.

~~~
from fibonacci_cyt import fib_cyt

fib_cyt(10)
~~~
{: .language-python}

Upon running it, you can see that it works the same as our regular version. We will get into ways on how to speed up
the code itself shortly.

> ## Compiling an addition module
>
> Define a simple addition module below, which containing the following function, and write it to a file called 
> `cython_addition.pyx.` Modify it to return x + y.
>
> ~~~
> def addition(x, y):
>    # TODO
> ~~~
> {: .language-python}
>
> Utilise the function by importing it into a new file, `addition_main.py`. Edit the `setup.py` accordingly to import
> the correct file. Use the demo above as a reference.
>
> > ## Solution
> >
> > `cython_addition.pyx`
> > ~~~
> > def addition(x, y):
> > print(x + y)
> > ~~~
> > {: .language-python}
> > 
> > `addition_main.py`
> > ~~~
> > from cython_addition import addition 
> > addition(2,3)
> > ~~~
> > {: .language-python}
> >
> > `setup.py`
> > ~~~
> > from distutils.core import setup, Extension
> > from Cython.Build import cythonize
> > 
> > setup(ext_modules = cythonize("cython_addition.pyx"))
> > ~~~
> > {: .language-python}
> > 
> > ~~~
> > $ python setup.py build_ext --inplace
> > $ python addition_main.py
> > ~~~
> > {: .language-bash}
> {: .solution}
{: .challenge}




{% include links.md %}
