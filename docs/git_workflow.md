# Git workflow

`main` is a small landing branch containing only `README.md`. Put all working
code on `deploy`. `scripts/boost_git.py` provides previews by default and never
force-pushes.

Add the BOOST repository as its own remote. The current `origin` may point at a
PATH-G1 repository, so do not reuse it for BOOST unless that is intentional.

```bash
git remote add boost <BOOST-repository-url>

# Preview, then publish the current folder and move this worktree to deploy.
python scripts/boost_git.py deploy --remote boost
python scripts/boost_git.py deploy --remote boost --apply

# One-time setup (or repair) of the README-only main branch.
python scripts/boost_git.py main-readme --remote boost
python scripts/boost_git.py main-readme --remote boost --apply
```

The `deploy` command stages all non-ignored changes, creates a commit when
needed, pushes to `boost/deploy`, and switches the current worktree to the local
`deploy` branch. It refuses a non-fast-forward update. `main-readme` creates a
normal forward commit whose resulting branch tree contains exactly `README.md`;
it does not rewrite branch history or force-push. The repo `.gitignore` keeps
external checkouts, local environments, and generated data out of the upload.
