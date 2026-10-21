#!/usr/bin/env python3
"""Publish the BOOST working tree to deploy and keep main README-only.

All commands are previews by default. Add --apply to perform a commit and push.
The remote must be named explicitly so the PATH-G1 origin cannot be selected by
accident.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LANDING_README = (
    "# BOOST\n\n"
    "Research code for object-centric teleoperation and manipulation.\n\n"
    "Active development is on the `deploy` branch.\n"
)


def git(
    repo: Path,
    *args: str,
    env: dict[str, str] | None = None,
    input_data: str | bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_data,
        text=not isinstance(input_data, bytes),
        capture_output=True,
        env=merged_env,
        check=False,
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result


def ensure_repo(repo: Path) -> Path:
    root = Path(git(repo, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if root != repo.resolve():
        raise RuntimeError(f"--repo must be the repository root, got {repo}")
    return root


def ensure_remote(repo: Path, remote: str) -> str:
    result = git(repo, "remote", "get-url", remote, check=False)
    if result.returncode:
        raise RuntimeError(
            f"Git remote '{remote}' is not configured. Add the BOOST repository first, "
            f"for example: git remote add {remote} <BOOST-repository-url>."
        )
    return result.stdout.strip()


def remote_branch_sha(repo: Path, remote: str, branch: str) -> str | None:
    result = git(repo, "ls-remote", "--heads", remote, f"refs/heads/{branch}")
    for row in result.stdout.splitlines():
        fields = row.split()
        if len(fields) == 2 and fields[1] == f"refs/heads/{branch}":
            return fields[0]
    return None


def fetch_branch(repo: Path, remote: str, branch: str) -> str:
    git(repo, "fetch", "--no-tags", remote, f"refs/heads/{branch}")
    return git(repo, "rev-parse", "FETCH_HEAD").stdout.strip()


def worktree_for_branch(repo: Path, branch: str) -> Path | None:
    output = git(repo, "worktree", "list", "--porcelain").stdout
    wanted = f"refs/heads/{branch}"
    for block in output.strip().split("\n\n"):
        path = None
        ref = None
        for line in block.splitlines():
            if line.startswith("worktree "):
                path = Path(line.removeprefix("worktree ")).resolve()
            elif line.startswith("branch "):
                ref = line.removeprefix("branch ")
        if ref == wanted:
            return path
    return None


def point_local_branch(repo: Path, branch: str, commit: str) -> None:
    existing = git(repo, "rev-parse", "--verify", f"refs/heads/{branch}", check=False)
    if existing.returncode == 0:
        existing_sha = existing.stdout.strip()
        if existing_sha != commit:
            backup = f"archive/{branch}-before-boost-publish-{existing_sha[:12]}"
            backup_result = git(repo, "show-ref", "--verify", f"refs/heads/{backup}", check=False)
            if backup_result.returncode == 0:
                backup_sha = backup_result.stdout.split()[0]
                if backup_sha != existing_sha:
                    raise RuntimeError(
                        f"Cannot preserve local {branch}: backup branch {backup} already exists."
                    )
            else:
                git(repo, "branch", backup, existing_sha)
            print(f"Preserved the previous local {branch} at {backup}.")
    git(repo, "branch", "-f", branch, commit)


def status_count(repo: Path) -> int:
    return len(git(repo, "status", "--porcelain=v1").stdout.splitlines())


def show_plan(repo: Path, remote: str, mode: str) -> None:
    current = git(repo, "branch", "--show-current").stdout.strip() or "(detached HEAD)"
    remote_url = ensure_remote(repo, remote)
    print(f"Repository: {repo}")
    print(f"Remote:     {remote} ({remote_url})")
    print(f"Branch:     {current}")
    print(f"Changes:    {status_count(repo)} path(s)")
    if mode == "deploy":
        print("Plan:       commit the current non-ignored folder, push to deploy (no force),")
        print("            then switch this worktree to the local deploy branch.")
    else:
        print("Plan:       publish a normal commit to main whose complete tree is README.md only.")
    print("Dry run:    no files, refs, commits, or remote branches changed; pass --apply to execute.")


def publish_deploy(repo: Path, remote: str, message: str) -> None:
    current = git(repo, "branch", "--show-current").stdout.strip()
    if not current:
        raise RuntimeError("Cannot publish from detached HEAD; check out a development branch first.")
    if current == "main":
        raise RuntimeError("main is reserved for the README landing page; publish from a development branch.")

    other_worktree = worktree_for_branch(repo, "deploy")
    if other_worktree is not None and other_worktree != repo.resolve() and current != "deploy":
        raise RuntimeError(f"Local deploy is checked out in another worktree: {other_worktree}")

    remote_sha = remote_branch_sha(repo, remote, "deploy")
    if remote_sha:
        remote_sha = fetch_branch(repo, remote, "deploy")
        ancestor = git(repo, "merge-base", "--is-ancestor", remote_sha, "HEAD", check=False)
        if ancestor.returncode:
            raise RuntimeError(
                "The remote deploy branch is not an ancestor of the current branch. "
                "Reconcile the branches manually; this CLI never force-pushes."
            )

    git(repo, "var", "GIT_AUTHOR_IDENT")
    git(repo, "var", "GIT_COMMITTER_IDENT")
    git(repo, "add", "-A")
    staged = git(repo, "diff", "--cached", "--quiet", check=False)
    if staged.returncode == 1:
        git(repo, "commit", "-m", message)
    elif staged.returncode != 0:
        raise RuntimeError("Could not inspect staged changes.")
    else:
        print("No changes to commit; publishing the current commit.")

    commit = git(repo, "rev-parse", "HEAD").stdout.strip()
    git(repo, "push", remote, f"{commit}:refs/heads/deploy")
    git(
        repo,
        "fetch",
        "--no-tags",
        remote,
        f"+refs/heads/deploy:refs/remotes/{remote}/deploy",
    )

    if current != "deploy":
        point_local_branch(repo, "deploy", commit)
        git(repo, "switch", "deploy")
    git(repo, "branch", "--set-upstream-to", f"{remote}/deploy", "deploy")
    print(f"Published {commit[:12]} to {remote}/deploy; this worktree is now on deploy.")


def make_readme_commit(repo: Path, parent: str | None, message: str) -> str:
    with tempfile.TemporaryDirectory(prefix="boost-main-index-") as temp_dir:
        index_path = str(Path(temp_dir) / "index")
        index_env = {"GIT_INDEX_FILE": index_path}
        git(repo, "read-tree", "--empty", env=index_env)
        blob_result = git(
            repo,
            "hash-object",
            "-w",
            "--stdin",
            input_data=LANDING_README.encode("utf-8"),
        )
        blob = blob_result.stdout.decode("ascii").strip()
        git(
            repo,
            "update-index",
            "--add",
            "--cacheinfo",
            f"100644,{blob},README.md",
            env=index_env,
        )
        tree = git(repo, "write-tree", env=index_env).stdout.strip()
        args = ["commit-tree", tree]
        if parent:
            args.extend(["-p", parent])
        args.extend(["-m", message])
        return git(repo, *args).stdout.strip()


def publish_main_readme(repo: Path, remote: str, message: str) -> None:
    current_branch = git(repo, "branch", "--show-current").stdout.strip()
    if current_branch == "main":
        raise RuntimeError("Switch this worktree off main before publishing the README-only branch.")
    git(repo, "var", "GIT_AUTHOR_IDENT")
    git(repo, "var", "GIT_COMMITTER_IDENT")
    parent = remote_branch_sha(repo, remote, "main")
    if parent:
        parent = fetch_branch(repo, remote, "main")
        current_tree = git(repo, "rev-parse", f"{parent}^{{tree}}").stdout.strip()
    else:
        current_tree = None

    commit = make_readme_commit(repo, parent, message)
    new_tree = git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()
    if current_tree == new_tree:
        print(f"{remote}/main already contains only the configured README; no push needed.")
        target_commit = parent
    else:
        names = git(repo, "ls-tree", "-r", "--name-only", commit).stdout.splitlines()
        if names != ["README.md"]:
            raise RuntimeError(f"Refusing to publish main with unexpected files: {names}")
        git(repo, "push", remote, f"{commit}:refs/heads/main")
        git(
            repo,
            "fetch",
            "--no-tags",
            remote,
            f"+refs/heads/main:refs/remotes/{remote}/main",
        )
        target_commit = commit

    checked_out = worktree_for_branch(repo, "main")
    if checked_out is None:
        point_local_branch(repo, "main", target_commit)
        git(repo, "branch", "--set-upstream-to", f"{remote}/main", "main")
    else:
        print(f"Remote main updated; local main is in use at {checked_out}, so its ref was left alone.")
    if current_tree != new_tree:
        print(f"Published README-only tree {commit[:12]} to {remote}/main (no force-push).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="repository root (default: this checkout)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("deploy", "main-readme"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--remote", required=True, help="configured Git remote name, such as boost")
        sub.add_argument("--apply", action="store_true", help="perform the commit/ref update and push")
        sub.add_argument("--message", default=None, help="commit message")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        repo = ensure_repo(args.repo.resolve())
        ensure_remote(repo, args.remote)
        if not args.apply:
            show_plan(repo, args.remote, args.command)
            return 0
        remote_url = ensure_remote(repo, args.remote)
        print(f"Target repository: {remote_url}")
        if args.command == "deploy":
            publish_deploy(repo, args.remote, args.message or "Publish BOOST working tree")
        else:
            publish_main_readme(repo, args.remote, args.message or "Keep main README-only")
        return 0
    except (OSError, RuntimeError) as error:
        print(f"boost_git: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
