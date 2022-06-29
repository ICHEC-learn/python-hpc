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

{% include links.md %}
