---
title: "MPI with Python"
teaching: 10
exercises: 0
questions:
- "What is the difference between a process and a thread?"
- "What is the difference between shared and distributed memory?"
- "What is a communicator and rank?"
objectives:
- "Write a simple 'Hello World' program with MPI"
- "Use Send and Receive to create a ping-pong program"
keypoints:
- "MPI is the true way to achieve parallelism"
- "`mpi4py` is an unofficial library that can be used to implement MPI in Python"
- "A communicator is a group containing all the processes that will participate in communication"
- "A rank is a logical ID number given to a process, and therefore a way to query the rank"
- "Point to Point communication is the communication between two processes, where a source sends a message to a 
  destination process which has to then receive it"
---

<p align="center"><img src="../fig/ICHEC_Logo.jpg" width="40%"/></p>


Now that we have finished with the cffi material it is time to get cracking into MPI, but before we do, we need to remind ourselves on the difference between threads and processes. If you recall yesterday when I was discussing about threads. It can be thought in simplified terms as hands on a keyboard.

One hand deals with one side of the keyboard, the other hand the other side, each of these can be thought of as an individual process. If we focus on one hand, each finger has specific tasks and keys which it should press, each of these can be thought of as a thread. Normally one works with either a group of processes or a group of threads in reality.

So the more official definitions;

What is a thread? READ

What is a process? READ

MPI works with processes, each process can be considered as a core on a machine. Most desktop machines nowadays have between 4 and 8 cores, so therefore, we can use MPI to make use of the cores. Using MPI is the true way to achieve parallelism.

But what is MPI? READ

I mentioned distributed memory, hands up if anyone has heard of distributed memory?

Distributed memory (READ)

As architecture trends changed, shared memory systems were combined over networks creating hybrid distributed memory / shared memory systems. MPI implementors adapted their libraries to handle both types of underlying memory architectures seamlessly. They also adapted/developed ways of handling different interconnects etc

So nowadays, MPI runs on virtually any hardware platform

So how do we go about running MPI in Python? Thankfully is it a simpler process than C or Fortran, as we normally need an MPI call to start the MPI library. In python, this isn’t needed as importing the library is enough.

What I also need to say is that MPI is not really very good in notebooks and I only do it here for demonstration purposes. When you come to do exercises later you will need to use job scripts and submit them to the queue. Although MPI can work on login nodes, just don’t do it. Always submit to the queue.

When submitting a job script which contains python scripts and MPI inside them we need to run the files differently.

We can’t do python3 my_file.py, mpi will not work. We need specific instructions, which I will get to later. When it comes to writing an MPI program you need three key things, the first is a  communicator

A communicator is a group containing all the processes that will participate in communication, done using an object MPI.COMM_WORLD. Without this, MPI is not possible, it is like the vocal chords of MPI.

Next is a rank, each rank starting from 0 will be associated with a process. It provides a way to query the process in the way that we all have names (READ)

The final thing that’s needed is the size, how big is our communicator. We specify this when we actually run the program, not here.

So every MPI program in python, after importing it will need these three things.

Because MPI can get complicated, keep to thesis naming conventions otherwise it will be confusing to you. If you have different communicators, you may need different naming conventions, but that’s for another time

Now. the simplest MPI program you can run.

(READ)

Go to exercise 1, modify the job script, and then submit a python file. I am aware of time, so we will do the exercises in small chunks, one and two together then 3 and 4

Point to Point communication, think of it like a game of ping pong (READ)

There are a few types of sends, if you want more details go to a dedicated C/Fortran course. We have a synchronous send, where the sender gets information that the message has been received, and an asynchronous send, where the sender knows that the message has left, but after that, it has no clue.

Here we have a simple send recieve we have our communicator

We start with our rank 0, which contains some data, in this case a dictionary, and we want to send this data, to a destination, and we specify the destination rank. If you have used MPI with C and Fortran this is a simplified notation.

Now, we want to say that if our rank equals 1, we have to receive it, for every send in this way of doing this we need a receive, specifying the rank from which the message came from. If Adam is rank0 and he sent me a letter to me at rank 1, I need to know to expect a message from him. If I am told to expect a message instead from Fionnuala, who is rank 2 here, I will be waiting until the end of time for her message to arrive. The message she never even sent! These are known as deadlocks

If we run this with 2 processes, it will work fine. What happens if we run with 4?

Exercise 2 is a bit tricky, so I can help out a bit more


left_rank = (rank - 1) % size - how does it relate to size, and if you are a rank, what would you think is left and right?


right_rank = (rank + 1) % size

Then for a selection of ranks, think about how you would determine your neighbours.
You will need separate sends to sned your rank to the ne on the left, and the one on the right, then define variables left and right to receive from others. Then for odd ranks, big hint there, how would you change that?

Parallel sum (READ)

Now we move onto communicating numpy arrays, and this is where you need to be careful. There are two sends and receives, one is capitalised, the other is not. Also the data needs to exists in the receive, previously all we needed to say is that there is data coming from another rank

So here we are sending an array of floats of size 100, and we need a buffer, we call data here to receive that array into

Before the exercises, I will chat a bit about combined sends and receives. This 

{% include links.md %}
