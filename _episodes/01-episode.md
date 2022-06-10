---
title: "Episode title"
teaching: 10
exercises: 0
questions:
- "How are these files formatted"
objectives:
- "Outline information sources"
- "How to import images, code snippets and abstractions"
- "How to structure callouts, highlighting"
keypoints:
- "Lesson episodes are stored in _episodes/dd-subject.md."
- "Each episode's title must include a title, time estimates, motivating questions, lesson 
  objectives, and key points."
- "Episodes should not use sub-titles or HTML layout."
- "Code blocks can have the source, regular output, or error class."
- "Special sections are formatted as blockquotes that open with a level-2 header and close with a
  class identifier."
- "Special sections may be callouts or challenges; other styles are used by the template itself."
---

<p align="center"><img src="../fig/ICHEC_Logo.jpg" width="40%"/></p>

## Overview

Carpentries lessons are written in standard markdown format with a few modifications, which will be
discussed in this episode.

For a full guide on Carpentries rationale, refer to the 
[documentation](https://carpentries.github.io/curriculum-development/designing-challenges.html)

A well written, clean Carpentries style code should not exceed 120 characters per line, otherwise
it could cause build issues with the tests. Refer to the lesson check python file 
[here](https://github.com/ICHEC-learn/ichec-template/blob/gh-pages/bin/lesson_check.py)

These `.md` should be viewed as supporting material to the slides that you are presenting, based
off the material in here. 

Most of the syntax is covered below, aside from a few other requirements which are non-standard.

- **Pictures**: See the implementation of the logo above for adding images and also their location.
- **Code snippets**: Instead of writing out a whole code and clogging up the contents of a markdown
                     file, snippets can be separate files stored in `_includes/snippets_library`

~~~
{% include {{ site.snippets }}/ep01/bash_commands.snip %}
~~~
{: .source}

- **Abstractions**: Carpentries comes with a handy way of abstracting commonly repeated references.
                    These are stored in the 
                    [`_config.yml`](https://github.com/ICHEC-learn/ichec-template/blob/gh-pages/_config.yml)

~~~
{{ site.remote.name }} is hosted by the {{ site.remote.location }}

{{ site.sched.submit }} myjob.sh
~~~
{: .source}

- **Jumping between episodes**: The next episode can be referred to with the arrow, but the most
                                common is 
                                [next lesson]({{page.root}}{% link _episodes/02-deep-thought.md %})

- **Data files**: If you are working on a course that requires the users to have some data, you can use links to the
                  [file]({{page.root}}{% link files/mydata.txt %}) in this example.

### Syntax Highlighting

The following styles like `.source`, but include syntax highlighting for the
specified language.
Please use them where possible to indicate the type of source being displayed,
and to make code easier to read.

`.language-bash`: Bash shell commands:

~~~
echo "Hello World"
~~~
{: .language-bash}

`.language-html`: HTML source:

~~~
<html>
<body>
<em>Hello World</em>
</body>
</html>
~~~
{: .language-html}

`.language-make`: Makefiles:

~~~
all:
    g++ main.cpp hello.cpp -o hello
~~~
{: .language-make}

`.language-matlab`: MATLAB source:

~~~
disp('Hello, world!')
~~~
{: .language-matlab}

`.language-python`: Python source:

~~~
print("Hello World")
~~~
{: .language-python}

`.language-r`: R source:

~~~
cat("Hello World")
~~~
{: .language-r}

`.language-sql`: SQL source:

~~~
CREATE PROCEDURE HelloWorld AS
PRINT 'Hello, world!'
RETURN (0)
~~~
{: .language-sql}

> ## Highlighting for other languages
> You may use other `language-*` classes to activate syntax highlighting
> for other languages.
> For example,
>
> {% raw %}
>     ~~~
>     title: "YAML Highlighting Example"
>     description: "This is an example of syntax highlighting for YAML."
>     array_values:
>         - value_1
>         - value_2
>     ~~~
>     {: .language-yaml }
> {% endraw %}
>
>
> will produce this:
>
> ~~~
> title: "YAML Highlighting Example"
> description: "This is an example of syntax highlighting for YAML."
> array_values:
>     - value_1
>     - value_2
> ~~~
> {: .language-yaml }
>
>
> Note that using `.language-*` classes other than
> `.language-bash`
> `.language-html`,
> `.language-make`,
> `.language-matlab`,
> `.language-python`,
> `.language-r`,
> or `.language-sql`
> will currently cause one of the tests in the lesson template's
> `make lesson-check` to fail for your lesson,
> but will not prevent lesson pages from building and rendering correctly.
>
{: .solution }


## Special Blockquotes

We use blockquotes to group headings and text
rather than wrapping them in `div` elements.
in order to avoid confusing [Jekyll][jekyll]'s parser
(which sometimes has trouble with Markdown inside HTML).
Each special blockquote must started with a level-2 header,
but may contain anything after that.
For example,
a callout is formatted like this:

~~~
> ## Callout Title
>
> text
> text
> text
>
> ~~~
> code
> ~~~
> {: .source}
{: .callout}
~~~
{: .source}

(Note the empty lines within the blockquote after the title and before the code block.)
This is rendered as:

> ## Callout Title
>
> text
> text
> text
>
> ~~~
> code
> ~~~
> {: .source}
{: .callout}

The [lesson template]({{ site.template_repo }}) defines styles
for the following special blockquotes:

<div class="row">
  <div class="col-md-6" markdown="1">

> ## `.callout`
>
> An aside or other comment.
{: .callout}

> ## `.challenge`
>
> An exercise.
{: .challenge}

> ## `.checklist`
>
> Checklists.
{: .checklist}

> ## `.discussion`
>
> Discussion questions.
{: .discussion}

> ## `.keypoints`
>
> Key points of an episode.
{: .keypoints}

  </div>
  <div class="col-md-6" markdown="1">

> ## `.objectives`
>
> Episode objectives.
{: .objectives}

> ## `.prereq`
>
> Prerequisites.
{: .prereq}

> ## `.solution`
>
> Exercise solution.
{: .solution}

> ## `.testimonial`
>
> A laudatory quote from a user.
{: .testimonial}

  </div>
</div>

Note that `.challenge` and `.discussion` have the same color but different icons.
Note also that one other class, `.quotation`,
is used to mark actual quotations
(the original purpose of the blockquote element).
This does not add any styling,
but is used to prevent the checking tools from complaining about a missing class.

Most authors will only use `.callout`, `.challenge`, and `.prereq`,
as the others are automatically generated by the template.
Note that `.prereq` is meant for describing things
that learners should know before starting this lesson;
setup instructions do not have a particular style,
but are instead put on the `setup.md` page.

Note also that solutions are nested inside exercises as shown below:

~~~
> ## Challenge Title
>
> This is the body of the challenge.
>
> ~~~
> it may include some code
> ~~~
> {: .source}
>
> > ## Solution
> >
> > This is the body of the solution.
> >
> > ~~~
> > it may also include some code
> > ~~~
> > {: .output}
> {: .solution}
{: .challenge}
~~~
{: .source}

The double indentation is annoying to edit,
but the alternatives we considered and discarded are worse:

1.  Use HTML `<div>` elements for the challenges.
    Most people dislike mixing HTML and Markdown,
    and experience shows that it's all too easy to confuse Jekyll's Markdown parser.

2.  Put solutions immediately after challenges rather than inside them.
    This is simpler to edit,
    but clutters up the page
    and makes it harder for tools to tell which solutions belong to which exercises.

## IMPORTANT

**`include links.md` surrounded by Liquid tags must be the last line in every file**

{% include links.md %}
