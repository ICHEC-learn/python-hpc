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

{% include links.md %}
