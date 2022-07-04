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
{: .callout}



{% include links.md %}
