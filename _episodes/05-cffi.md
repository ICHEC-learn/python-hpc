---
title: "C Foreign Function Interface for Python"
teaching: 10
exercises: 0
questions:
- "What is a foreign function interface?"
- "How can I implement CFFI?"
- "Under what circumstances should I use CFFI?"
objectives:
- "Understand the rationale of a foreign function interface"
- "Implement up to 4 different styles of using CFFI"
keypoints:
- "CFFI is an external package for Python that provides a C Foreign Function Interface for Python, and allows one to 
  interact with almost any C code"
- "The Application Binary Interface mode (ABI) is easier, but slower"
- "The Application Programmer Interface mode (API) is more complex but faster"
---

<p align="center"><img src="../fig/ICHEC_Logo.jpg" width="40%"/></p>

## Foreign Function Interfaces

As we know there are advantages and disadvantages for coding in higher level languages, be it readability or 
performance related, and one must take into account that a lot of libraries which can be very useful are written in
these lower level languages, and if you only work in a higher level language, and the utility you need is in a lower 
level language, then you have the tendency to get stuck, but there are a few things you can do, whether it be;

- Porting some or all of the library to your language of choice
- Write an extension in C to that library to bridge the gap between the library and your language (in this case, Python)
- Wrap the library in your preferred language’s foreign function interface.

This is not just exclusive to Python, as there FFI libraries in languages like Ruby, Lisp, Perl etc, and FFI wrappers 
are generally easier to write and maintain then a C-extension and are more portable and usable. The FFI term itself 
refers to the ability for code written in one language, known as the host, in our case here being python to access and 
invoke functions from another guest language, in this case being C, but you can also use fortran, NumPy has a `f2py` 
library as part of its functionality.

It sounds like a great solution, but just because you can do something, doesn’t always mean its the best idea, 
particularly if;

1. **You have a lower level library that you have made yourself or another one which has been optimised.** This is 
   particularly the case if that lower level library processes arrays at a fast rate using threading or multiple 
   processes, it is suited to that language and you are going to get the best use of it in that environment.
2. **There are delicate callbacks from the guest language into the host.** Lower level languages can have very 
   specific callbacks which can cause problems in higher level languages if not called correctly.
3. **If your lower level library makes use of compile time or preprocessor features.** Pure python does not have this 
   functionality.


## C Foreign Function Interface for Python - `cffi`

Let us look more into the `cffi` library itself, which as the name suggests is the C foreign function interface for
Python. It only applies to C, so C++ cannot be used here.

To utilise CFFI well, a working knowledge of C is recommended. Although the Cython material from the
[previous episode](https://ichec-learn.github.io/python-hpc/04-cython/index.html) is a good introduction of the 
concepts, some practical experience with C will help for your learning experience.

The concept of CFFI is that you add C-like declarations to the Python code, so it has its similarities to Cython, but
with some differences. In Cython we looked at static type declarations declaring variables as `int` or `double` etc, in
CFFI, we will be dealing with items in the C source code such as pointers, which are a fundamental concept in C but not 
referred to in python, so we need to declare these variables in Python to make sure they exist. We will give an 
introduction to pointers and get you to use them in upcoming exercises.

> ## Difference between Windows and Mac systems
>
> There is the potential to become stuck here due to the naming conventions of files depending whether you are on a 
> Windows, Mac or UNIX system. Library files on Windows and UNIX systems tend to have the extension `.dylib`, whereas
> on Macs, they can have extensions of `.so` or in some cases `.so.6`.
>
> This is why CFFI, for the majority of cases is best saved for personalised code sets rather than widely used publicly
> available code, as 
>
{: .callout}

## CFFI Modes

CFFI has two different modes which it operates from, **ABI** and **API**, and we will look first at ABI.

**ABI - Application Binary Interface**

This mode accesses the guest library at the binary level, which sounds ideal as all code gets translated into binary 
code, but here, it can have major difficulties if you have non-Windows setup. Even so it is like playing with fire, 
because the function calls themselves need to go through the `libffi` library which is a slow step and a significant 
performance bottleneck. Despite this, it is considered to be the easier of the two, modes s

**API - Application Programmer Interface**

This method contains a separate compilation step with a C compiler. Due to the use of the C compiler, this method is
recommended, allied with the fact it is also faster. 

Each of these modes have sub-modes out-of-line and in-line. This refers to the format of how you write the necessary 
code.

Before we dive into the modes themselves, we need to cover an important topic in C, pointers.

## Pointers in C

This is a complex topic at the best of times but when dealing with code in C, it is ineviteable that you will encounter
pointers. Put simply, a pointer is a variable that contains the address of a variable. The reason they are used is
because sometimes they are the only way of expressing a computation, and they can lead to more compact and efficient
code.

Let's take a simplified version of what pointers actually do, as the concept can be tricky for non-C users. Say you
are working your way through a computation, in the way that a zookeeper is working through their enclosure records. 
Imagine you are sitting in an office, and the amount of space in your office is equivalent to the memory available in
your computer program. You can bring some insects and small furry animals to catalog them without much difficulty, but
if you keep them in the room whilst you are creating your catalog (or running your program), you are going to run out 
of memory space fairly quickly. Similarly, if you brought an elephant into your room... everyone knows the phrase! 
Think of the animals as individual variables, or the elephant as a very large one. Importing and having variables in
your program can waste a lot of memory. 

So in this situation, how is the zookeeper going to know and keep track of what animals he/she actually has? In this
example, let's say they set up a video link to every enclosure in the zoo. The video link displays on it the name of
the enclosure and the zookeeper can see what animals are present. You can think of this as the enclosure *address*. 
Rather than getting the animals into an enclosed space one by one or multiple ones at the same time, all you need is
some method to identify where the animal actually is, and you can still do some work based on that. 

This is the concept of what a pointer does, containing the address of a variable, or in tune with our example, the
enclosure name of "Nellie the Elephant". 

The address in computational terms is an integer variable, and it is the physical location in memory to where a 
variable is stored. When dealing with pointers, we often do something known as **deferencing**, which is the act of
referring to where the pointer points to, rather than the memory address. When we access the nth element of an array,
we are doing the same thing. Arrays themselves are technically pointers, accessing the first item in an array is the
same as referencing a pointer. In C this is done using the `*` operator. 


Let's have a look at some code to see how it works. There are two methods of dealing with pointers.

~~~
#include <stdio.h>

int main()
{
    /* We define a local variable A, the actual declaration */
    int A = 1;
    /* Now we define the pointer variable, and point it to A using the & operator */ 
    int * pointer_to_A = &A;

    printf("Method 1:\nThe value A is %d\n", A); 
    printf("The value of A is also %d\n", *pointer_to_A);

    int B = 2;
    /* Let's define a pointer variable pB */
    int * pB; 
    /* store the address of B in pointer variable */ 
    pB = &B;

    printf("\nMethod 2:\nThe address of B variable is %p\n", &B); 
    printf("However the address stored in the pB variable is %p\n", pB); 
    printf("And the value of the *pB variable is %d\n", *pB);

    return 0;
}
~~~
{: .language-c}

~~~
$ gcc -o pointers.o pointers.c
$ ./pointers.o
~~~
{: .language-bash}

~~~
Method 1:
The value A is 1
The value of A is also 1

Method 2:
The address of B variable is 0x7ffee11d36bc
However the address stored in the pB variable is 0x7ffee11d36bc
And the value of the *pB variable is 2
~~~
{: .output}

> ## Practice with pointers
> 
> Take some time to fiddle with the code above, see what works, how it fails to compile at times and why. Try writing
> your own code this time with `double` for variables, similar to the above until you get used to it. Will the `double`
> type work with the pointers?
>
{: .challenge}

## Importing an existing library

So how to use it?

Let's looks into how to import an existing library and a particular function in that library, here we use the square 
root function in the C math library.

~~~
from cffi import FFI
ffi = FFI()

lib = ffi.dlopen("libm.so.6")
ffi.cdef("""float sqrtf(float x);""")
a = lib.sqrtf(4)
print(a)
~~~
{: .language-python}

1. We import the FFI class from cffi and set it equal to a variable ffi. This top level class is instantiated once and
   only needed once per module.
2. From there we can use the newly defined `ffi` variable to open the library using the `dlopen` function, which 
   returns a dynamic library, in this case the `libm` library from C. 
3. Now we use `ffi.cdef` to define the function we want, we only need the declaration of the function not the 
   implementation. The library file itself will have its own declaration of what this function does and we don’t need
   to worry about that, all we need in this case is the the declaration of the function. This will cause python to 
   recognise that the `sqrtf` can be used and we can now use our variable lib, which we have defined to open the
   library and call the function.

> ## Library files across different operating systems
>
> Library files have a `.so` extension for Windows and UNIX systems and `.dylib` for Macs. So if you are a Mac user and
> wish to get this working locally, ensure that you replace the `.so` with `.dylib`. Depending on your machine on Linux
> or Windows system, an additional `.6` extension may or may not be needed. Usually if `.so` does not work, then the
> `.so.6` extension will.
>
{: .callout}

## Creating and using your own library

Now for the fun stuff, creating and using your own library. Of the 4 combinations that CFFI has, we will look at the
**ABI in-line** method.

### ABI in-line

The first step is to create a file in C, which we will call `ABI_add.c`, a very simple C file that adds two variables.

~~~
#include <stdio.h>

void add(double *a, double *b, int n)
{
    int i;
    for (i=0; i<n; i++) {
        a[i] += b[i];
    }
}
~~~
{: .language-c}

Next we create a `Makefile`. These are handy tools, however many programmers can’t write one of these from scratch and 
borrow one which someone else has made earlier. We won't cover this in much detail, but what it is doing is compiling
the C file and putting it into a `.so` library.

~~~
CC=gcc
CFLAGS=-fPIC -O3
LDFLAGS=-shared

mylib_add.so: ABI_add.c
	$(CC) -o ABI_add.so $(LDFLAGS) $(CFLAGS) ABI_add.c
~~~
{: .language-bash}

From here we can create our library by typing `make` in the terminal. Have a quick check of the directory contents
before running the command using `ls` and see how it has changed.

~~~
$ make
~~~
{: .language-bash}

You can see if you type `ls` again that our library has been added as a `.so` file.

Now we need to go about importing the library and casting our variables and this is where we can start working with
Python again. There are a few specific functions to get the C library which we created working in Python.

- `ffi.dlopen(libpath, [flags])` - opens and returns a handle to a dynamic library, which we can then use later to 
  call the functions from that library
- `ffi.cdef("C-type", value)` - creates a new ffi object with the declaration of function
- `ffi.cast("C-type", value)` - The value is casted between integers or pointers of any type
- `ffi.from_buffer([cdecl], python_buffer)` - Return an array of cdata that points to the data of the given Python 
  object

Let's look at the code for importing and running this library:

~~~
from cffi import FFI
import numpy as np
import time 

ffi = FFI()
lib = ffi.dlopen('./ABI_add.so')
ffi.cdef("void add(double *, double *, int);")


t0=time.time()
a = np.arange(0,200000,0.1)
b = np.ones_like(a)

# "pointer" objects
aptr = ffi.cast("double *", ffi.from_buffer(a))
bptr = ffi.cast("double *", ffi.from_buffer(b))

lib.add(aptr, bptr, len(a))
print("a + b = ", a)
t1=time.time()

print ("\ntime taken for ABI in line", t1-t0)
~~~
{: .language-python}

Once we have imported our library, we use the `ffi.cdef` to declare our `add` function. But as we saw in the original
C code, the input is two pointers. Python doesn’t know what to do with them, so we have two variables that python can’t
handle by itself, so we need to cast them, using the `ffi.cast` function. We can call them `aptr` and `bptr` as we can 
then associate them for what they are. From there, we are telling python that the value of the `double *` that we refer
to in the C code, is actually going to be this `ffi.from_buffer` variable we have defined.

Then we can use our `lib` handle to call the addition function using `aptr` and `bptr`. 

That's how to work with ABI in-line, now let's look at the other method. **API out-of-line**.

### API out-of-line

Our first step is to create a python "build" file. This is similar to how we worked with Cython. We need to set up an
`ffi.builder` handle, then using `ffi.cdef` create a new ffi-object. We will also introduce the `ffibuilder.set_source`
function which gives the name of the extension module to produce, and we input some C source code as a string as an
argument. This C code needs to make the declared functions, types and globals available. The code below represents
what we are putting into our file which we will call `API_add_build.py`

~~~
import cffi
ffibuilder = cffi.FFI()
ffibuilder.cdef("""void API_add(double *, double *, int);""")
ffibuilder.set_source("out_of_line._API_add", r"""
  void API_add(double *a, double *b, int n)
  {
      int i;
      for (i=0; i<n; i++){
          a[i] += b[i];
          }
          /* or some algorithm that is seriously faster in C than in Python */
  }
  """)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
~~~
{: .language-python}

When we set a variable called `ffibuilder` to `cffi.FFI()`, the difference with the ABI version is more of a syntax 
checker and making sure you don’t get mixed up between the different modes you use.

When we use the `set_source` function, the implementation goes in. First we define the name of the module, `out_of_line`
and then the extension `._API_add`. This will be be module folder and then the module name, `API_add`. The `r` 
indicates we want the function to read the following C code.

You may wonder what happens if you have a long C function. You can add in a header file and it will read that instead, 
feel free to try that out for yourselves. 

At the end we have a compile function that runs through a C compiler.

Now we run our build file in the terminal. This will create a new directory, called `out_of_line`, which is the name of
our module. Inside that directory we will have our library file (`.so`/`.dylib`) and our `.c` and `.o` files.

~~~
$ python API_add_build.py
~~~
{: .language-bash}

From here we import the library and like before cast the variables.

~~~
import numpy as np 
import time
from out_of_line._API_add import ffi, lib

t0=time.time()
a = np.arange(0,200000,0.1)
b = np.ones_like(a)

# "pointer" objects
aptr = ffi.cast("double *", ffi.from_buffer(a))
bptr = ffi.cast("double *", ffi.from_buffer(b))

lib.API_add(aptr, bptr, len(a))
print("a + b = ", a)
t1=time.time()
print("\ntime taken for API out of line", t1-t0)
~~~
{: .language-python}

It is the same structure as as the previous method, import the library and cast the variables. The only one difference
here compared to the ABI version is in the import, where we import the library directly from `out_of_line._API_add` as
well as `ffi`.

> ## Possible errors 
>
> You may find a ImportError related to dynamic module export function. If this happens, try working on your code in a
> different directory.
>
{: .callout}

> ## Fibonacci example
>
> Consider the fibonacci code given in [`fibonacci.c`](../files/04-CFFI/exercise1_2/fibonacci.c). It requires the user 
> to input a positive integer `n`, which is the number of terms to be performed by the `fibo` function, for which the 
> arguments are two `int *`.
>
> The `fibo` function itself is utilised in a `for` loop.
>
> Use either ABI in-line or API out-of-line methods as outlined above to import and implement this code.
> 
> Below is a skeleton code you can use to implement your library.
> 
> ~~~
> import numpy as np 
> import time
> 
> # TODO (API) From newly created library import ffi, lib
> 
> # TODO (ABI) Open library and define function
> 
> # Number of terms in sequence
> n = 10
> 
> # TODO: Define your pointer objects (HINT: there is no buffer and nothing to cast. Create a new variable using ffi.new)
> aptr = 
> bptr = 
> 
> # Sets up the first two terms of fibonacci
> aptr[0] = 0
> bptr[0] = 1
> 
> for i in range(n+1):
>     # TODO: Call the function
>     print(bptr[0]
> ~~~
> {: .language-python}
>
> > ## Solution
> > 
> > A full solution can be found in the Jupyter notebook [here](../files/04-CFFI/soln/04-Soln-cffi.ipynb).
> > 
> {: .solution}
{: .challenge}

> ## Evolve
>
> This exercise is based on the `evolve.py` file which we have used a few times during this course. You can implement
> this in either ABI or API modes, or both!
> 
> All existing file names are linked. You may wish to create a few copies of 
> [`heat_equation_simple.py`](../files/04-CFFI/exercise3/heat_equation_simple.py) for different methods.
>
> ### API mode
>
> By copying and pasting the C code in [`evolve.c`](../files/04-CFFI/exercise3/evolve.c), create a `build` file to 
> utilize the C code using API out-line mode.
>
> - Run the build file to create the library
> - Import your newly created library back into 
>   [`heat_equation_simple.py`](../files/04-CFFI/exercise3/heat_equation_simple.py)
>
> ### ABI mode
>
> The files [`evolve.h`](../files/04-CFFI/exercise3/evolve.h) and [`evolve.c`](../files/04-CFFI/exercise3/evolve.c) 
> contain a pure C implementation of the single time step in the heat equation. The C implementation can be built into
> a shared library with the provided [`Makefile`](../files/04-CFFI/exercise3/Makefile) by executing the make command.
>
> - Edit the heat_equation_simple.py file to use CFFI in the ABI in-line mode.
> - Utilize the library function instead of the Python function.
> 
> > ## Solution
> >
> > A full solution can be found in the Jupyter notebook [here](../files/04-CFFI/soln/04-Soln-cffi.ipynb).
> {: .solution}
{: .challenge}

{% include links.md %}