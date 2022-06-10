# Instructions for creating new repositories

This repository should intentionally remain as a private repo, so therefore is not published on an external github.io page. 
Upon following the instructions below, the repo will automatically run integration tests, spelling checks and page building tests. 

The only folders and files therein which need to be edited for lesson use are as follows;

- `_episodes/` or `episodes_rmd/`: Episodes either in regular Markdown or RMarkdown
- `_includes/snippets_library/`: Including code snippets (text files, outputs etc.) This is for convenience, as it can be easily 
  written in markdown
- `_config.yml`: As above, specific commands/abstractions which can be used 
- `fig/`: Insert figures
- `data/`: Any data needed for the duration of the lesson
- `slides/`: Any powerpoint/pdf slides being used for the lesson
- `files`, `code`: Can be utilised as above subject to content

Any files which are not included in the above should be tested in the `ichec-template` repository before continuing.

To create a new repository from this template and to ensure you have your own copy, follow the instructions outlined below:

1. Select **Use this template** 
2. Select **ICHEC-learn** as the owner then call the repo something memorable (e.g. intro-to-linux), and give a description
  (can be edited later)
3. Set to **Public** and make sure the final **Include all branches** is **NOT** unticked.
4. The new repo will be created and the tests will run automatically. 
5. Enable `github.io` page building through Settings -> Pages. You may need to set the theme first time. The theme choice will not matter as 
  it will be forced into the Carpentries style. 
6. Copy the link and paste into About, on the right hand side of the main page.
7. Next head up to **Fork**, which allows you to freely experiment with the repo in your own account without affecting the original project.
8. From here you can now work on Github online or locally (via cloning). Repeat steps 5 & 6 from the previous to have your own publishable github.io page.

Working Online/Locally

1. Online: Upon editing the file it needs to be committed, similar to saving. Ensure you commit to the `gh-pages` branch. **NB:** If 
   there are other contributors, ensure that on your forked repo, you click **Fetch upstream** regularly.
2. If you are creating a local copy on your machine;
    a) Head to the **Code** button and copy the link, then clone locally by typing `git clone mylink.git`
    b) Edit as normal, to commit changes;
        - `git add myfile`
        - `git commit -m "my changes"`
    c) To push your local changes to git, simply type `git push`
    d) **NB**: If changes have been made by others and you have fetched on your forked repo in github (step 1 above), ensure you type
       `git fetch` to merge your local laptop changes with your forked online repo (which has been synced with the ICHEC-learn main repo)

```
________________________________________________________________________
|                                                                       |
| Remote                             ICHEC-learn/lesson-repo            |
|                                       |                ^              |
|                                       |                |              |
|---------------------------------------v----------------|--------------|
|                                Fetch upstream    Merge Requests       |
|---------------------------------------|----------------|--------------|
|                       |               v                |              |
| Local (fork)          | Online       username/lesson-repo             |
|                       |               |                ^              |
|                       |---------------v----------------|--------------|
|                       |          `git fetch`      `git push`          |
|                       |---------------|-------------------------------|
|                       | Laptop        |          `git commit`         |
|                       |               |                ^              |
|                       |               v                |              |
|                       |             `/path/to/lesson-repo`            |
|_______________________|_______________________________________________|
```

To merge your changes with the main ICHEC-learn;

1. On your forked repo, `{yourusername}/{repo-name}` head to the top bar and select **Pull requests** 
2. Head to the right and select **New pull request**
3. On the next screen, check it is able to merge and **Create pull request**
4. Fill in the Title of your pull request (Comments below are optional but recommended for reviewers) and then **Create pull request**
5. Wait for the tests to finish (~5 mins) at which point, the *All checks have passed* message will appear and;
    - contact a fellow colleague working on the repo and ask them to review.
    - or if you are the sole contributor 
6. Press the **Merge pull request**, then **Confirm merge**

**IMPORTANT**

If you are working off a repo with multiple contributions, ensure you commit and push daily

If a change has been merged to the main-repo in a file which you are working on, pull requests cannot automatically be merged,
but you can still create the pull request.

You then need to head to **Resolve conflicts**. At this point you will need to resolve the areas where `>>>>>>>>>>gh-pages` and 
`========` is displayed, otherwise it won't be resolved. This work will not save automatically, so has to be solved in one sitting.
