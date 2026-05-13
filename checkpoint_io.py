"""Shared I/O helpers for checkpoint files.

Used by both the training/inference entry point (`gpt.py`) and the
standalone downloader (`download_checkpoints.py`).
"""

import hashlib
import json
import subprocess


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def git_sha() -> str | None:
    """Short HEAD SHA with `-dirty` suffix if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None
    if subprocess.run(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL).returncode:
        sha = f"{sha}-dirty"
    return sha


def manifest_repo() -> str | None:
    """Read the HF Hub repo from checkpoints.json if present."""
    try:
        with open("checkpoints.json") as f:
            return json.load(f).get("repo")
    except Exception:
        return None


def upload_checkpoint(local_path: str, profile: str, step: int,
                      git_sha: str, repo: str) -> str:
    """Upload `local_path` to the given HF Hub model repo, renaming the
    remote file to `ckpt_<profile>_step_<step>_<sha>.pt`. Creates the
    repo if it doesn't exist. Returns the remote filename on success."""
    from huggingface_hub import create_repo, upload_file  # pyright: ignore[reportUnknownVariableType]
    remote = f"ckpt_{profile}_step_{step:05d}_{git_sha}.pt"
    create_repo(repo, repo_type="model", exist_ok=True)
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote,
        repo_id=repo,
        repo_type="model",
    )
    return remote
