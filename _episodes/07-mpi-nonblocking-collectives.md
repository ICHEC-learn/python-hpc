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

{% include links.md %}
