"""Download published checkpoints listed in checkpoints.json.

  uv run python download_checkpoints.py                # all
  uv run python download_checkpoints.py --step 4999    # just one
  uv run python download_checkpoints.py --list         # show manifest

Files land in ./checkpoints/ named `ckpt_step_<step>.pt` so the rest of
the codebase (training, inference, viz) finds them unchanged.

Pure stdlib — no extra deps required. The URLs in the manifest are
plain HTTPS; works regardless of which host you eventually use.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

MANIFEST = "checkpoints.json"
DEST = "checkpoints"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _progress(name: str):
    def hook(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0:
            pct = min(100, done * 100 // total_size)
            print(
                f"\r  {name}: {done / 1e6:6.1f} / {total_size / 1e6:6.1f} MB ({pct}%)",
                end="",
                flush=True,
            )
    return hook


def download_one(entry: dict) -> None:
    fname = f"ckpt_step_{entry['step']:05d}.pt"
    dest = os.path.join(DEST, fname)
    expected = entry.get("sha256")

    if os.path.exists(dest):
        if expected and sha256_file(dest) == expected:
            print(f"  {fname}: already present, sha256 ok")
            return
        print(f"  {fname}: exists but sha256 differs — re-downloading")

    print(f"  {fname}: downloading from {entry['url']}")
    urllib.request.urlretrieve(entry["url"], dest, reporthook=_progress(fname))
    print()

    if expected:
        got = sha256_file(dest)
        if got != expected:
            os.remove(dest)
            sys.exit(
                f"\nERROR sha256 mismatch for {fname}:\n"
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
    entries = manifest["checkpoints"]

    if args.list:
        for e in entries:
            print(f"  step {e['step']:>5}  sha {e['git_sha']:>8}  {e['url']}")
        return

    if args.step is not None:
        entries = [e for e in entries if e["step"] == args.step]
        if not entries:
            sys.exit(f"no checkpoint with step {args.step} in {MANIFEST}")

    os.makedirs(DEST, exist_ok=True)
    print(f"downloading {len(entries)} checkpoint(s) to {DEST}/")
    for e in entries:
        download_one(e)


if __name__ == "__main__":
    main()
