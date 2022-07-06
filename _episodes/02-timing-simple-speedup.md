---
title: "Timing Code and Simple Speed-up Techniques"
teaching: 10
exercises: 0
questions:
- "How can I time my code"
- "What is the difference between vectorisation and for loops?"
- "What is the cache?"
- "How can `lru_cache` speed up my code?"
objectives:
- "Introduce `time`, `timeit` modules for timing code and `cProfile`, `pstats` for code profiling"
keypoints:
- "Performance code profiling is used to identify and analyse the execution and improvement of applications."
- "Never try and optimise your code on the first try. Get the code correct first."
- "Most often, about 90% of the code time is spent in 10% of the application."
- "The `lru_cache()` helps reduce the execution time of the function by using the memoization technique, discarding
   least recently used items first."
---

<p align="center"><img src="../fig/ICHEC_Logo.jpg" width="40%"/></p>

> ## Using Jupyter Notebook cell magics
> 
> Magics are specific to and provided by the IPython kernel. They can be particularly useful in notebooks and activate
> certain utilities. There are different activations one can use. To save code written in a code block, you can use the
> following notation, for a simplified Python file.
> 
> ~~~
> %%writefile hello.py
> 
> print("Hello World!")
> ~~~
> {: .language-python}
>
> This will save the code block as a file called `hello.py`. In a Jupyter notebook cell, you can type in `%` followed
> by `Tab` and all the normal cell magics will be displayed.
> 
> We will be implementing cell magics shortly with the `timeit` module.
{: .callout}

{% include links.md %}
