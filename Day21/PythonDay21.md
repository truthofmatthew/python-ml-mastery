# Advanced Git for Python Developers

Git is a crucial tool for Python developers, providing version control and facilitating collaboration on projects. This tutorial will delve into advanced Git commands and workflows that can enhance your productivity and effectiveness in team projects.

## Task 1: Create and Manage Branches in a Git Repository

Branching in Git allows you to diverge from the main line of development and continue to work without messing up the live project. Here's how to create and manage branches:

1. **Create a new branch**: Use the command `git branch branch_name` to create a new branch. Replace `branch_name` with your desired name.

```bash
git branch feature_branch
```

2. **Switch to the new branch**: Use `git checkout branch_name` to switch to your newly created branch.

```bash
git checkout feature_branch
```

3. **Make changes and commit**: Make your changes in this branch and commit them using `git commit -m "commit message"`.

```bash
git commit -m "Add new feature"
```

4. **Merge changes back to the main branch**: First, switch back to the main branch using `git checkout main`. Then, use `git merge branch_name` to merge the changes from your branch to the main branch.

```bash
git checkout main
git merge feature_branch
```

## Task 2: Rebase Your Branch onto the Main Branch

Rebasing is another way to integrate changes from one branch into another. It involves moving or combining a sequence of commits to a new base commit.

1. **Switch to the branch you want to rebase**: Use `git checkout branch_name`.

```bash
git checkout feature_branch
```

2. **Rebase onto the main branch**: Use `git rebase main`.

```bash
git rebase main
```

If there are any conflicts, Git will pause and allow you to resolve those conflicts before continuing. Once resolved, you can continue the rebase with `git rebase --continue`.

## Task 3: Utilize Git Stash to Save Uncommitted Changes

`git stash` is a powerful command that allows you to save changes that you have not committed yet, so you can apply them later.

1. **Stash your changes**: If you have made changes that you are not ready to commit yet, you can stash them. Use `git stash save "stash message"`.

```bash
git stash save "work in progress for new feature"
```

2. **Apply stashed changes**: You can apply the stashed changes to your current working branch with `git stash apply`.

```bash
git stash apply
```

Remember, mastering Git commands and workflows is essential for effective collaboration in Python projects. Practice these commands to become proficient in managing and integrating changes across different branches.