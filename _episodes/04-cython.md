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


Define a simple addition module below, which containing the following function, and write it to a file cython_addition.pyx. Modify it to return x + y.
def addition(x, y):
    # TODO
Utilise the function by importing it into a new file addition_main.py
Edit the setup.py accordingly to import the correct file. Use the demo as a reference
Try the exercise outside the notebook environment using a text editor like vim or nano or another editor of your choosing
%%writefile cython_addition.pyx
​
## TODO: Copy the above code into this cell and modify it to return x + y
def addition(x, y):
    print(x + y)
Writing cython_addition.pyx
%%writefile addition_main.py
​
## TODO: Import the function
from cython_addition import addition 
## TODO: Call the function
addition(2,3)
Writing addition_main.py
%%writefile setup.py
​
from distutils.core import setup, Extension
from Cython.Build import cythonize
​
# TODO: Edit the next line
setup(ext_modules = cythonize("cython_addition.pyx"))
Writing setup.py
!python setup.py build_ext --inplace
Compiling cython_addition.pyx because it changed.
[1/1] Cythonizing cython_addition.pyx
/ichec/home/staff/fisolomon/.conda/envs/python_hpc/lib/python3.7/site-packages/Cython/Compiler/Main.py:369: FutureWarning: Cython directive 'language_level' not set, using 2 for now (Py2). This will change in a later release! File: /ichec/home/staff/fisolomon/hpc-python/py-hpc-2022-05/03-Cython/exercise/cython_addition.pyx
  tree = Parsing.p_module(s, pxd, full_module_name)
running build_ext
building 'cython_addition' extension
gcc -pthread -B /ichec/home/staff/fisolomon/.conda/envs/python_hpc/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/ichec/home/staff/fisolomon/.conda/envs/python_hpc/include/python3.7m -c cython_addition.c -o build/temp.linux-x86_64-3.7/cython_addition.o
gcc -pthread -shared -B /ichec/home/staff/fisolomon/.conda/envs/python_hpc/compiler_compat -L/ichec/home/staff/fisolomon/.conda/envs/python_hpc/lib -Wl,-rpath=/ichec/home/staff/fisolomon/.conda/envs/python_hpc/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/cython_addition.o -o build/lib.linux-x86_64-3.7/cython_addition.cpython-37m-x86_64-linux-gnu.so
copying build/lib.linux-x86_64-3.7/cython_addition.cpython-37m-x86_64-linux-gnu.so -> 
!python addition_main.py
5
Exercise 2: 5 minutes
Consider the following python code which computes the integral for the function 𝑥2−𝑥
Run the code as is and record the time, 𝑡1
Use Cython's static type declarations on variables only and record the new time
Repeat, and implement function call overheads. How has the time, 𝑡1 changed?
Remember to use both cdef and cpdef
%load_ext cython
%%cython -a 
​
def f(x):
    return x ** 2 - x
​
def integrate_f(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx
Generated by Cython 0.29.28

Yellow lines hint at Python interaction.
Click on a line that starts with a "+" to see the C code that Cython generated for it.

 01: 
+02: def f(x):
/* Python wrapper */
static PyObject *__pyx_pw_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_1f(PyObject *__pyx_self, PyObject *__pyx_v_x); /*proto*/
static PyMethodDef __pyx_mdef_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_1f = {"f", (PyCFunction)__pyx_pw_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_1f, METH_O, 0};
static PyObject *__pyx_pw_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_1f(PyObject *__pyx_self, PyObject *__pyx_v_x) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("f (wrapper)", 0);
  __pyx_r = __pyx_pf_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_f(__pyx_self, ((PyObject *)__pyx_v_x));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_f(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_x) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("f", 0);
/* … */
  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_AddTraceback("_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a.f", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}
/* … */
  __pyx_tuple_ = PyTuple_Pack(1, __pyx_n_s_x); if (unlikely(!__pyx_tuple_)) __PYX_ERR(0, 2, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple_);
  __Pyx_GIVEREF(__pyx_tuple_);
/* … */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_1f, NULL, __pyx_n_s_cython_magic_5afef5ba8fb36a6f24); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 2, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_f, __pyx_t_1) < 0) __PYX_ERR(0, 2, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_codeobj__2 = (PyObject*)__Pyx_PyCode_New(1, 0, 1, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple_, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_ichec_home_staff_fisolomon_cach, __pyx_n_s_f, 2, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__2)) __PYX_ERR(0, 2, __pyx_L1_error)
+03:     return x ** 2 - x
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyNumber_Power(__pyx_v_x, __pyx_int_2, Py_None); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 3, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyNumber_Subtract(__pyx_t_1, __pyx_v_x); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 3, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_r = __pyx_t_2;
  __pyx_t_2 = 0;
  goto __pyx_L0;
 04: 
+05: def integrate_f(a, b, N):
/* Python wrapper */
static PyObject *__pyx_pw_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_3integrate_f(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_3integrate_f = {"integrate_f", (PyCFunction)(void*)(PyCFunctionWithKeywords)__pyx_pw_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_3integrate_f, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_3integrate_f(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_a = 0;
  PyObject *__pyx_v_b = 0;
  PyObject *__pyx_v_N = 0;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("integrate_f (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_a,&__pyx_n_s_b,&__pyx_n_s_N,0};
    PyObject* values[3] = {0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_a)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_b)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("integrate_f", 1, 3, 3, 1); __PYX_ERR(0, 5, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_N)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("integrate_f", 1, 3, 3, 2); __PYX_ERR(0, 5, __pyx_L3_error)
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "integrate_f") < 0)) __PYX_ERR(0, 5, __pyx_L3_error)
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 3) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
    }
    __pyx_v_a = values[0];
    __pyx_v_b = values[1];
    __pyx_v_N = values[2];
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("integrate_f", 1, 3, 3, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 5, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a.integrate_f", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_2integrate_f(__pyx_self, __pyx_v_a, __pyx_v_b, __pyx_v_N);
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_2integrate_f(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_N) {
  PyObject *__pyx_v_s = NULL;
  PyObject *__pyx_v_dx = NULL;
  PyObject *__pyx_v_i = NULL;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("integrate_f", 0);
/* … */
  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_AddTraceback("_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a.integrate_f", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF(__pyx_v_s);
  __Pyx_XDECREF(__pyx_v_dx);
  __Pyx_XDECREF(__pyx_v_i);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}
/* … */
  __pyx_tuple__3 = PyTuple_Pack(6, __pyx_n_s_a, __pyx_n_s_b, __pyx_n_s_N, __pyx_n_s_s, __pyx_n_s_dx, __pyx_n_s_i); if (unlikely(!__pyx_tuple__3)) __PYX_ERR(0, 5, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__3);
  __Pyx_GIVEREF(__pyx_tuple__3);
/* … */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_46_cython_magic_5afef5ba8fb36a6f24f5d5f1cf4d614a_3integrate_f, NULL, __pyx_n_s_cython_magic_5afef5ba8fb36a6f24); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 5, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_integrate_f, __pyx_t_1) < 0) __PYX_ERR(0, 5, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
+06:     s = 0
  __Pyx_INCREF(__pyx_int_0);
  __pyx_v_s = __pyx_int_0;
+07:     dx = (b - a) / N
  __pyx_t_1 = PyNumber_Subtract(__pyx_v_b, __pyx_v_a); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 7, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_PyNumber_Divide(__pyx_t_1, __pyx_v_N); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 7, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_dx = __pyx_t_2;
  __pyx_t_2 = 0;
+08:     for i in range(N):
  __pyx_t_2 = __Pyx_PyObject_CallOneArg(__pyx_builtin_range, __pyx_v_N); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 8, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  if (likely(PyList_CheckExact(__pyx_t_2)) || PyTuple_CheckExact(__pyx_t_2)) {
    __pyx_t_1 = __pyx_t_2; __Pyx_INCREF(__pyx_t_1); __pyx_t_3 = 0;
    __pyx_t_4 = NULL;
  } else {
    __pyx_t_3 = -1; __pyx_t_1 = PyObject_GetIter(__pyx_t_2); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 8, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_4 = Py_TYPE(__pyx_t_1)->tp_iternext; if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 8, __pyx_L1_error)
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  for (;;) {
    if (likely(!__pyx_t_4)) {
      if (likely(PyList_CheckExact(__pyx_t_1))) {
        if (__pyx_t_3 >= PyList_GET_SIZE(__pyx_t_1)) break;
        #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_2 = PyList_GET_ITEM(__pyx_t_1, __pyx_t_3); __Pyx_INCREF(__pyx_t_2); __pyx_t_3++; if (unlikely(0 < 0)) __PYX_ERR(0, 8, __pyx_L1_error)
        #else
        __pyx_t_2 = PySequence_ITEM(__pyx_t_1, __pyx_t_3); __pyx_t_3++; if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 8, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_2);
        #endif
      } else {
        if (__pyx_t_3 >= PyTuple_GET_SIZE(__pyx_t_1)) break;
        #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_2 = PyTuple_GET_ITEM(__pyx_t_1, __pyx_t_3); __Pyx_INCREF(__pyx_t_2); __pyx_t_3++; if (unlikely(0 < 0)) __PYX_ERR(0, 8, __pyx_L1_error)
        #else
        __pyx_t_2 = PySequence_ITEM(__pyx_t_1, __pyx_t_3); __pyx_t_3++; if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 8, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_2);
        #endif
      }
    } else {
      __pyx_t_2 = __pyx_t_4(__pyx_t_1);
      if (unlikely(!__pyx_t_2)) {
        PyObject* exc_type = PyErr_Occurred();
        if (exc_type) {
          if (likely(__Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration))) PyErr_Clear();
          else __PYX_ERR(0, 8, __pyx_L1_error)
        }
        break;
      }
      __Pyx_GOTREF(__pyx_t_2);
    }
    __Pyx_XDECREF_SET(__pyx_v_i, __pyx_t_2);
    __pyx_t_2 = 0;
/* … */
  }
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
+09:         s += f(a + i * dx)
    __Pyx_GetModuleGlobalName(__pyx_t_5, __pyx_n_s_f); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 9, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __pyx_t_6 = PyNumber_Multiply(__pyx_v_i, __pyx_v_dx); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 9, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __pyx_t_7 = PyNumber_Add(__pyx_v_a, __pyx_t_6); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 9, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_7);
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    __pyx_t_6 = NULL;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_5))) {
      __pyx_t_6 = PyMethod_GET_SELF(__pyx_t_5);
      if (likely(__pyx_t_6)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_5);
        __Pyx_INCREF(__pyx_t_6);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_5, function);
      }
    }
    __pyx_t_2 = (__pyx_t_6) ? __Pyx_PyObject_Call2Args(__pyx_t_5, __pyx_t_6, __pyx_t_7) : __Pyx_PyObject_CallOneArg(__pyx_t_5, __pyx_t_7);
    __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
    __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
    if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 9, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_2);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_5 = PyNumber_InPlaceAdd(__pyx_v_s, __pyx_t_2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 9, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_DECREF_SET(__pyx_v_s, __pyx_t_5);
    __pyx_t_5 = 0;
+10:     return s * dx
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyNumber_Multiply(__pyx_v_s, __pyx_v_dx); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 10, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;
%time integrate_f(9, 10, 100)
CPU times: user 177 µs, sys: 103 µs, total: 280 µs
Wall time: 290 µs
80.74334999999998
%%cython
​
## TODO: redo previous code with static type declarations & function call overheads
def f(double x):
    return x ** 2 - x
​
def integrate_f(double a, double b, int N):
    cdef int s
    cdef double dx
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx
## TODO: record time of cython version, using %time (remember to use cpdef with %time)
%time integrate_f(9, 10, 100)
CPU times: user 48 µs, sys: 28 µs, total: 76 µs
Wall time: 103 µs
80.74334999999998
%%cython
​
## TODO: redo previous code with static type declarations & function call overheads
cdef int f_cdef(int x):
    return x ** 2 - x
​
cpdef double integrate_fcdef(double a, double b, int N):
    cdef int s
    cdef double dx
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_cdef(a + i * dx)
    return s * dx
## TODO: record time of cython version, using %time (remember to use cpdef with %time)
%time integrate_fcdef(9, 10, 100)
CPU times: user 0 ns, sys: 43 µs, total: 43 µs
Wall time: 63.7 µs
72.0
Additional Exercises
Optimising with Cython
Based on a cProfile obtained for the simple (inefficient) heat equation script, write the setup.py file that creates a Cython module for the most time consuming part of the script. Below is the beginning of the cProfile for heat_equation_simple.py.

In its current state, running the script can be done using the following command either in the terminal or using ! from the notebook;

!python heat_equation_simple.py bottle.dat
Runtime: 20.442239999771118
Using the inefficient script on the current setup the runtime of heat_equation_simple.py using;
bottle.dat ≃ ~20 secs
bottle_large.dat ≃ ~10 mins
The time module is being used to check the runtime of the iterate function, which implements evolve. The actual runtime of heat_equation_simple.py is slightly longer.
How much can you improve the performance? You should be able to optimise it sufficiently to get the performance runtimes down to;
bottle.dat < 0.1 sec
bottle_large.dat < 5 secs
Can you get it even better?

NB: Experiment with bottle.dat first, and ignore bottle_large.dat otherwise it will take too long! Get bottle.dat to run for ~0.5 secs before attempting with the larger dataset.

Drawing
Use the code segment below from evolve.py to create an evolve.pyx file. If you wish, you can copy this into a text editor of your choosing.
Create a setup file to compile the code
Create the cython extension and investigate the effect on performance using any prefered timing technique.
Insert static type declarations, function calls, cnp arrays and compiler directives where necessary.
%%writefile evolve_cyt.pyx
​
import numpy as np
cimport numpy as cnp
import cython
​
​
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
​
def evolve(cnp.ndarray[cnp.double_t , ndim =2]u , cnp.ndarray[cnp.double_t , ndim =2]u_previous, double a, double dt, double dx2, double dy2):
    
    cdef int n = u.shape[0]
    cdef int m = u.shape[1]
​
    cdef int i,j
    #multiplication is more efficient than division
    cdef double dx2inv = 1. / dx2
    cdef double dy2inv = 1. / dy2
​
    for i in range(1, n-1):
        for j in range(1, m-1):
            u[i, j] = u_previous[i, j] + a * dt * ( \
             (u_previous[i+1, j] - 2*u_previous[i, j] + \
              u_previous[i-1, j]) *dx2inv + \
             (u_previous[i, j+1] - 2*u_previous[i, j] + \
                 u_previous[i, j-1]) *dy2inv )
    u_previous[:] = u[:]
Writing evolve_cyt.pyx
%%writefile heat_setup.py
​
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
​
​
setup(ext_modules = cythonize("evolve_cyt.pyx"),include_dirs=[numpy.get_include()])
Writing heat_setup.py
!python heat_setup.py build_ext --inplace
Remember to import the correct module, evolve is different from evolve_cyt!

!python heat_equation_simple.py bottle.dat
Runtime: 18.84291100502014
Bonus:
Compare with heat_equation_index.py which uses array operations derived from evolve_index.py
What additional speed up can be found here?
!python heat_equation_index.py bottle.dat
Running Time: 0.06571698188781738
%%writefile evolve_index_cyt.pyx
​
import numpy as np
cimport numpy as cnp
import cython
​
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
​
def evolve_index_cyt(cnp.ndarray[cnp.double_t , ndim =2]u, cnp.ndarray[cnp.double_t , ndim =2]u_previous, double a, double dt, double dx2, double dy2):
    
    del_sqrd_u = (u_previous[:-2, 1:-1] - 2*u_previous[1:-1, 1:-1] + u_previous[2:, 1:-1]) / dx2 + (u_previous[1:-1, :-2] - 2*u_previous[1:-1, 1:-1] + u_previous[1:-1, 2:]) / dy2 
    
    u[1:-1,1:-1] = u_previous[1:-1,1:-1] + dt * a *  del_sqrd_u
​
    u_previous[:,:] = u[:,:]
%%writefile heat_setup.py
​
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
​
​
setup(ext_modules = cythonize("evolve_index_cyt.pyx"),include_dirs=[numpy.get_include()])
!python heat_setup.py build_ext --inplace
!python heat_equation_index.py bottle.dat


{% include links.md %}
