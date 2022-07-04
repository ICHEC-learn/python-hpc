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

- Porting some or all of the library to your language of choice**
- Write an extension in C to that library to bridge the gap between the library and your language (in this case, Python)
- Wrap the library in your preferred language’s foreign function interface.

This is not just exclusive to Python, as there FFI libraries in languages like Ruby, Lisp, Perl etc, and FFI wrappers 
are generally easier to write and maintain then a C-extension and are more portable and usable. The FFI term itself 
refers to the ability for code written in one language, known as the host, in our case here being python to access and 
invoke functions from another guest language, in this case being C, but you can also use fortran, NumPy has a f2py 
library in there as well

It sounds like a great solution, but just because you can do something, doesn’t always mean its the best idea, 
particularly if;

1. **You have a lower level library that you have made yourself or another one which has been optimised.** This is 
   particularly the case if that lower level library processes arrays at a fast rate using threading or multiple 
   processes, it is suited to that language and you are going to get the best use of it in that environment.
2. **There are delicate callbacks from the guest language into the host.** Lower level languages can have very 
   specific callbacks which can cause problems in higher level languages if not called correctly.
3. **If your lower level library makes use of compile time or preprocessor features.** Pure python does not have this 
   functionality


## C Foreign Function Interface for Python - `cffi`

Let us look more into the `cffi` library itself, which as the name suggests is the C foreign function interface for
Python. It only applies to C, so C++ cannot be used here.

To utilise CFFI well, a working knowledge of C is recommended. Although the Cython material from the
[previous episode](04-cython.md) is a good introduction of the concepts, some practical experience with C will help
for your learning experience.

The concept of CFFI is that you add C-like declarations to the Python code, so it has its similarities to Cython, but
with some differences. In Cython we looked at static type declarations declaring variables as `int` or `double` etc, in
CFFI, we will be dealing with items in the C source code such as pointers, which are a fundamental concept in C but not 
referred to in python, so we need to declare these variables in Python to make sure they exist. We will give an 
introduction to pointers and get you to use them in upcoming exercises.

## Importing an existing library

We are 

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

### ABI - Application Binary Interface

This mode accesses the guest library at the binary level, which sounds ideal as all code gets translated into binary 
code, but here, it can have major difficulties if you have non-Windows setup. Even so it is like playing with fire, 
because the function calls themselves need to go through the `libffi` library which is a slow step and a significant 
performance bottleneck. Despite this, it is considered to be the easier of the two, modes s

2. API - Application Programmer interface and a separate compilation step with a C compiler is needed and is recommended as a result of that, as well as being faster. The work we will be doing here, you won’t see much of a speedup, its more to get you used to how the libraries work.

Worth mentioning that each of these modes have sub-modes out-of-line and in-line. This refers to the format of how you write the necessary code.

5:

SO here I will be going through two different setups, ABI in line and API out-of line.

The ffi interface is used like a library, and there is a few different functions you can use, the main ones are ffi.cast where the value of a pointer is cast into an integer, so you define a variable referring to a pointer

Ffi.dlopen which opens and returns an access point to the library you use.

Also ffibuilder, but its better to get to work actually using these functions

6:

Importing an existing library

So how to use it?

Lets looks into how to import an existing library and a particular function in that library, here we use the square root function in the c math library

We import the FFI class from cffi and set it equal to a variable ffi. This top level class is instantiated once or do it once per module 

We can use that ffi variable to open the library using the dlopen function, which returns a dynamic library. Library files have a .so extension for Windows and UNIX systems and .dylib for Macs, so if you are a Mac user and want to get this working on your own machine, Depending on the machine, the .6 extension may or may not be needed

Now we use ffi.cdef to define the function we want, we only need the declaration of the function not the implementation. The library file itself will have its own declaration of what this function does and we don’t need to worry about that, all we need in this case is the the declaration of the function. This will cause python to recognise that the sqrtf can be used and we can now use our variable lib, which we have defined to open the library and call the function

Now for the fun stuff, creating and using your own library. Here we are going to do things as simply as possible, the same thing twice.

7:

Step 1: Create a file in C



Our C file has this function in it, we don’t necessarily need a C file which has a main function, as we just want to use this particular function. 

8:

Step 2: Create a MakeFile

These can be a pain to write and a lot of programmers can’t write one of these from scratch and borrow one which someone else has made earlier, same concept here, 
CC is the C compiler we are using
CFLAGS the optimisation
Some more flags and then the command that runs them. So you needn’t worry about this, but this is the file we are creating

Step 3: using the make command in the terminal

We run it and see;

When we run make in the terminal it will then make the library for you. 

We see now that a .so file has been added here, and this .so file is our library

9:

Step 4: importing it

We create a new file which in the repo is ABI.py

Next we cast the variables, because in the add function, in the original C code the input is two pointers and python doesn’t know what to do with them. The pointers in the C code point to a location in memory at which a variable of a certain value is stored

Without going into details, we have two varaibles that python can’t handle by itself, so we need to cast them, we can call them aptr and bptr as we can then associate them for what they are. We are saying here that the value of the double * that we refer to in the C code, is actually going to be this ffi.from_buffer variable we have defined.

Then we can use our lib handle to call the addition function using aprt and bptr. That’s how its done.

It may seem confusing at first, that’s ok it was for me initially, but we have enough time for you to try this out. 

10

API

Same code we are trying to implement, different method. This time we are writing a build file in python.

first , import Cffi, then set a variable called ffibuilder to cffi.FFI(). Why the difference, more of a syntax and making sure you don’t get mixed up between the different modes you use.

Use cdef again like we did previously, define the function.

Now here things are different where we use the set_source function. This is where your function implementation goes in. First we define a the name of the module, out_of_line anf then dot _API_add. This will be be module folder and then the module name, API_add, after that we type r, indicting we want the function to read the following C code.

What if I have a long C function you might ask, good question, you can just add in a header file and it will read that, but for the moment this will serve our purpose well

At the end we have a compile function that runs through a C compiler. Like I said before, this is the better, albeit theoretically slightly trickier way to do it, but I personally find it easier.

Step 2

Now we run our python file and lets see what that looks like by typing ls, listing the contents of the directory.

The python build file has created an ou-of-line folder, and within it, the library .so file and the .c and .o files. 

11:

Step 3

Now same as before import the library and cast the variables, there is only one difference here compared to the last time we ran this and that is line 3 here. Instead of opening the library using dlopen and assigning it to a variable called lib, here we import the library directly from out_of_line._API_add as well as ffi

Our pointers stay the same, the rest of the code is as above.

12:

So here is a summary of the different methods you can use either using ABI or API modes.

Do you want me to go over the steps of this again briefly before we move onto an exercise?

Exercise

Now a few exercises for you to occupy yourselves with for the rest of the session, first check out the code in the demo folder then submit the job so you are comfortable with it, then try the fibonacci and evolve codes and see how you do.


{% include links.md %}
