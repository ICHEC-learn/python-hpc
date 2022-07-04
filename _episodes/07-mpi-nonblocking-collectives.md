---
title: "Non-blocking and collective communications"
teaching: 10
exercises: 0
questions:
- "Why do we need non-blocking communication?"
- "What are the different collective routines and how can I implement them?"
objectives:
- "Understand the difference between blocking and non-blocking communication"
- "Implement different collective routines"
keypoints:
- "In some cases, serialisation is worse than a deadlock, as you don't know the code is inhibited by poor performance"
- "Collective communication transmits data among all processes in a communicator, and must be called by all processes
  in a group"
- "MPI is best used in C and Fortran, as in Python some function calls are either not present or are inhibited by poor
  performance"
---

<p align="center"><img src="../fig/ICHEC_Logo.jpg" width="40%"/></p>

Good afternoon everyone, today we are finishing off MPI before moving onto Dask later on.

Before we start please make sure that you are logged in, you need to copy over a new notebook, which contained spoilers yesterday, hence it wasn’t added immediately, please type

cp /ichec/home/users/course00/05-MPI/MPI-Part2.ipynb .

I’ll give you a minute to do that so you can follow along yourselves.

Yesterday we covered the basics of MPI, becoming familiar with communicators, rank, size, running mpi for simple calculations and then onto sending and receiving. 

We are going to continue with sends and receives, but a type where we can eliminate the risk of deadlocks, among other things.

We have covered small s send, r recv, for python and big S Send, R Recv, however these are blocking routines.

Lets take a case where we have a cyclic structure, where all our processes form part of a ring, hre we have rank 0’s neighbours being rank 1 and rank 3, and so on then we also have a non cyclic workflow

If we try and program this with MPI Send and Recv, for small messages this is can work, but you are playing with fire.. Because as soon as MPI switches to a synchronous protocol, then all the processes stop in MPI_Send, and Recv is never called, and all are Blocked forever. For non-cyclic, there is an upside, the last process is not calling send, so one will start recv, then the next rank will start receive.

So if we make no improvements, for the moment, what do you think the more preferable situation is?

Deadlock is better in this circumstance, as you can easily identify it. The non-cyclic situation is worse, as you have no indication that your code is being slowed down by a factor of 3. What if you had 10 processes, 50, 100? Imagine if you have 1000 processes in 3 dimensions, 9 connections

For non blocking communication we need to split it into 3 phases.
-Initiate nonblocking communication
-Returns immediately
- then have our return value as a request object
-They start with I meaning immediately or incomplete and are done locally, returning independelty of any other processes activity
-Do some work, other communication. Most important usage area is to overlap comms with comms. Have several comms that use the MPI library in parallel
-Wait for nonblocking communication to complete
-The send buffer is read out of recv is readlly read in
-
So, you do your send, like sending an email, you can then do other stuff until you hear a bling saying your email has been sent, at which point your send buffer is refreshed
Call Irecv, and do some work, wait until the message has arrived
Lets think of our example working with a ring.



– We call the non-blocking sends and these are coming back immediately, then we can call the blocking recv, so all processes are calling the blocking recv, so now the send is called in no-blocking way and the recv is called in a blocking way, which allows the message to flow, the call the MPI_Wait for the non-blocking send, but because the message has been received there is no need for extra waiting, and then you can reuse the send buffer

Or you can initiate a non-blocking receive,, each calling receive, then the other work we can do is a blocking send, the message can be transferred.

There is another example here, and this is a situation where one needs to know the boundary conditions in a process, in P0, we have technically our final row, the border_data and then some ghost data as well as the data taht stays local to the process.

We initialise a request, to Irecv the ghost data from process 1, and then Isend the border_data to process 1 so it can then do its own calculation. We can then do some work, as it is non-blocking, then wait for the sends and receives to complete, and then we can work on the border data

When it comes to waiting for data, the waitany and waitall methods are also handy

So onto a working example with a small array before you can have a little go of this yourselves. Note that we import the Request class up here

We set up our data which is a simple size 10 numpy array multiplied by the (rank + 1), and our receive buffer, where this will be put into. These now will exist in both rank 0 and 1.

On rank 0 it will be 0,1,2,3,4,5,6,7,8,9, and on rank 1, 0,2,4,6,8,10,12,14,16,18, and the buffer will still be an array of 0s


Create our request to send the data, then to our buffer, which is empty, we append the contents of the received message from process 1. 

Now exercise 5. Non-blocking communication is probably the most important topic in MPI, so if you get non-blocking communication, you are very well set for most problems you may encounter.

So I’ll brief you on this exercise before you get started

(READ)

This code can work, but it is still wrong, deceptively so. If you replace the Send with Ssend, a synchronous send, you are guaranteed to cause a deadlock. This example will run very quickly so if a result doesn’t come back in a matter of seconds, you have a deadlock.

From our exercise yesterday, we set up our left and right neighbours as well. Now for this exercise, use Issend rather than Isend. This will enable to to bring up deadlocks if they occur. You can use it for testing, but in reality, real applications would use Isend

For queues, sbatch the job script will submit the job, you can monitor the status of your job with squeue thejob id, and scancel to cancel the job. If you encounter a deadlock you must cancel the job I cannot stress this highly enough. If you are nervous about it, let us know in the chat and we will come to a breakout room to assist. 

10 minutes.

Collective communication

Collective communication is the step up from simple sends and receives, transmits data among all processes in a process group. Regardless of the group, these routines must be called by all processes, and you need to be careful about having the correct amount of sent and received data.

These operations can be used for data movement, collective computation or synchronization

So the idea is we have have something similar to original code like this to only one line.

We will cover three main types and then another different one before going into more detail about reductions, and you can then do exercises 5 and 6.
Broadcasting does exactly what it says on the tin, broadcasting the data from one process to all of them

Lets looks at a broadcasting file. We have our communicator, and rank, and we are broadcasting a python dictionary, and a numpy array to all the other processes

All the other processes have empty numpy arrays of the same size, as for the python object is doesn’t matter, but you do need to declare that the variable actually exists. Note the difference between the lowercase and uppercase again

Scatter

Now scattering is very similar to broadcasting, but the difference is important. MPI Broadacst, like the news sends the same information to all processes, whereas Scatter sends chunks of an array to different processes.

If anyone has watched BBC news, you have the main broadcast, think of like MPI Bcast, then you have the regional information, one for Northern Ireland, one for midlands, scotland etc. That is an example of scatter.

For lowercase scatter, we only need our python data and the root process, however for the numpy scatter, we need a receive buffer.

Gathering

Gathering is the inverse of Scatter Instead of spreading elements from one process to many processes, MPI_Gather takes elements from many processes and gathers them to one single process. This routine is highly useful to many parallel algorithms, such as parallel sorting and searching.

Using the combination of Scatter and Gather you can do so much, a very simple example being the computation of the average of numbers.

For the demonstration we want an list n that gets the gathered ranks, this is a simple python list. As with Scatter, Gather needs a buffer which we define here.

So as you can see these three have a lot in common and behave similarly.

There is an exercise which you can go into in which we will ge trying to substitute the Sends and Receives with a single Gather. I’ll give you 15 mins for this. If you finish it, you can use the time to catch up with some of the other exercises, and try different things out. You can also head down to past exercise 8, where there is a slightly easier exercise. You can work on A-D, and we will cover E in the next section.
15 mins

Now we are going to group this next section together, and have a larger exercise section which will lead into the break.

Reduce

Reduce is a classic concept from functional programming. Data reduction involves reducing a set of numbers into a smaller set of numbers via a function. For example, let’s say we have a list of numbers [1, 2, 3, 4, 5]. Reducing this list of numbers with the sum function would produce sum([1, 2, 3, 4, 5]) = 15. Similarly, the multiplication reduction would yield multiply([1, 2, 3, 4, 5]) = 120.
As you might have imagined, it can be very cumbersome to apply reduction functions across a set of distributed numbers. Along with that, it is difficult to efficiently program non-commutative reductions, i.e. reductions that must occur in a set order. Luckily, MPI has a handy function called MPI_Reduce that will handle almost all of the common reductions that a programmer needs to do in a parallel application.
So lets get an idea of what a sum reduction would do (READ)

What would the rank reduction be?

[0,1,2,3]
[0,2,4,6]
[0,3,6,9]
=[0,6,12,18]

READ

We specify the type of reduction that we want as well as the root rank, the operation op is optional, and defaults to MPI.SUM, which will return the sum. Alternatively you can use MPI.PROD for the product. Feel free to check it out.

Other collectives include (READ) 

There are also non blocking collectives in C and Fortran, but these are not supported in MPI4py to my knowledge

THese Non-blocking collectives enable the overlapping of communication and computation together with the benefits of collective communication.
But there are some problems:
Have to be called in same order by all ranks in a communicator.
Mixing of blocking and non-blocking collectives is not allowed
If you want to work with these types of operations, work with it in C or Fortran. To my knowledge this is also not available in C++, but you’ll have to double check the MPI standards for C++

Watch out for collectives as there can be some issues

Using a collective operation within one branch of an if-test of the rank.
if rank == 0: comm.bcast(...)
All processes in a communicator must call a collective routine!
Assuming that all processes making a collective call would complete at the same time.
Using the input buffer as the output buffer:
comm.Scatter(a, a, MPI.SUM)

Collective communications involve all the processes within a communicator.
All processes must call them.
Collective operations make code more transparent and compact.
Collective routines allow optimizations by MPI library.
MPI-3 also contains non-blocking collectives, but these are currently not supported by MPI for Python.
Documentation for mpi4py is quite limited
MPI used in C, C++, Fortran, and ideally not suited for python as a whole
If you are serious about MPI, we suggest utilising a different language of your choice
Leads to better performance as a result

Performance of mpi4py using for a ping-pong test
It is possible but not recommended to communicate arbitrary Python objects
NumPy arrays can be communicated with nearly the same speed as in C/Fortran


We will move onto communicators now, then have our exercise session

If you recall yesterday, we had our communicator MPI_COMM_WORLD, all processes that we want to involve in the communication, we can split up the processes to work in different communicators so they do different things. Like a couple weather model

We can split up these communicators as shown here. We have 8 ranks in MPI COMM world, but as you can see, as they go into the new communicators, their ranks reset from 0. 

Looking at the code you see we have a variable called colour. The value determining in which group the calling MPI process will be; MPI processes providing the same colour value will be put in the same subgroup.

Looking at the output, we have defined two sub communicators, each with their own local ranks

Now you can get on with the exercises. Up until just before 2:30, where I will round up MPI and we can have a break before heading into Dask


{% include links.md %}
