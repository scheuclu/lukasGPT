"""Download published checkpoints listed in checkpoints.json.

  uv run python download_checkpoints.py                # all
  uv run python download_checkpoints.py --step 4999    # just one
  uv run python download_checkpoints.py --list         # show manifest

Files land in ./checkpoints/ named `ckpt_step_<step>.pt` so the rest of
the codebase finds them unchanged. Downloads go through huggingface_hub
so we get caching, resume, and retries for free.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

MANIFEST = "checkpoints.json"
DEST = "checkpoints"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def download_one(repo: str, entry: dict) -> None:
    step = entry["step"]
    expected = entry.get("sha256")
    local = os.path.join(DEST, f"ckpt_step_{step:05d}.pt")

    if os.path.exists(local):
        if expected and sha256_file(local) == expected:
            print(f"  step {step:>5}: already present, sha256 ok")
            return
        print(f"  step {step:>5}: exists but sha256 differs — re-fetching")

    print(f"  step {step:>5}: fetching {entry['filename']} from {repo}")
    cached = hf_hub_download(repo_id=repo, filename=entry["filename"])
    shutil.copy(cached, local)

    if expected:
        got = sha256_file(local)
        if got != expected:
            os.remove(local)
            sys.exit(
                f"\nERROR sha256 mismatch for step {step}:\n"
                f"  expected {expected}\n"
                f"  got      {got}\n"
                f"  (file deleted)"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", type=int, default=None,
                    help="Download only the checkpoint at this step.")
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
            print(f"  step {e['step']:>5}  git {e['git_sha']:>8}  {e['filename']}")
        return

    if args.step is not None:
        entries = [e for e in entries if e["step"] == args.step]
        if not entries:
            sys.exit(f"no checkpoint with step {args.step} in {MANIFEST}")

    os.makedirs(DEST, exist_ok=True)
    print(f"downloading {len(entries)} checkpoint(s) from {repo} to {DEST}/")
    for e in entries:
        download_one(repo, e)


if __name__ == "__main__":
    main()
