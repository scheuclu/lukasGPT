"""Download published checkpoints listed in checkpoints.json.

  uv run python download_checkpoints.py                          # all
  uv run python download_checkpoints.py --step 4999              # all profiles, step 4999
  uv run python download_checkpoints.py --profile default        # all steps of one profile
  uv run python download_checkpoints.py --profile default --step 4999
  uv run python download_checkpoints.py --list                   # show manifest

Files land in ./checkpoints/ named `ckpt_<profile>_step_<step>.pt`.
Downloads go through huggingface_hub so we get caching, resume, and
retries for free.
"""

import argparse
import json
import os
import shutil
import sys
from typing import Any

from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

from checkpoint_io import sha256_file

MANIFEST = "checkpoints.json"
DEST = "checkpoints"


def download_one(repo: str, entry: dict[str, Any]) -> None:
    profile = entry["profile"]
    step = entry["step"]
    tag = f"{profile} step {step:>5}"
    expected = entry.get("sha256")
    local = os.path.join(DEST, f"ckpt_{profile}_step_{step:05d}.pt")

    if os.path.exists(local):
        if expected and sha256_file(local) == expected:
            print(f"  {tag}: already present, sha256 ok")
            return
        print(f"  {tag}: exists but sha256 differs — re-fetching")

    print(f"  {tag}: fetching {entry['filename']} from {repo}")
    cached = hf_hub_download(repo_id=repo, filename=entry["filename"])
    shutil.copy(cached, local)

    if expected:
        got = sha256_file(local)
        if got != expected:
            os.remove(local)
            sys.exit(
                f"\nERROR sha256 mismatch for {tag}:\n"
                f"  expected {expected}\n"
                f"  got      {got}\n"
                f"  (file deleted)"
            )


def main() -> None:
    description = (__doc__ or "").splitlines()[0] if __doc__ else None
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--profile", type=str, default=None,
                    help="Only download checkpoints from this hyperparameter profile.")
    ap.add_argument("--step", type=int, default=None,
                    help="Only download checkpoints at this step.")
    ap.add_argument("--list", action="store_true",
                    help="List available checkpoints and exit.")
    args = ap.parse_args()

    if not os.path.exists(MANIFEST):
        sys.exit(f"manifest {MANIFEST} not found in cwd")

    with open(MANIFEST) as f:
        manifest = json.load(f)
    repo = manifest["repo"]
    entries = manifest["checkpoints"]

    if args.list:
        print(f"repo: {repo}")
        for e in entries:
            print(f"  {e['profile']:>8}  step {e['step']:>5}  git {e['git_sha']:>8}  {e['filename']}")
        return

    if args.profile is not None:
        entries = [e for e in entries if e["profile"] == args.profile]
    if args.step is not None:
        entries = [e for e in entries if e["step"] == args.step]
    if not entries:
        criteria = []
        if args.profile is not None:
            criteria.append(f"profile={args.profile}")
        if args.step is not None:
            criteria.append(f"step={args.step}")
        sys.exit(f"no checkpoint matches {' '.join(criteria) or 'manifest'}")

    os.makedirs(DEST, exist_ok=True)
    print(f"downloading {len(entries)} checkpoint(s) from {repo} to {DEST}/")
    for e in entries:
        download_one(repo, e)


if __name__ == "__main__":
    main()
