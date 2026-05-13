"""Project Gutenberg books via DeepMind's PG-19 corpus.

PG-19 is ~28k books published before 1919, boilerplate already stripped
upstream. HF hosts only the manifest; the books themselves live on a
public GCS bucket. We download a configurable subset (default ~3000
books / ~2 GB, comparable to TinyStories) and concatenate them.
"""

import os
import re
import urllib.request
from collections import Counter

from huggingface_hub import (
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
)

from .base import Dataset

# DeepMind strips most license headers but footers sometimes leak through.
# Match common Gutenberg end markers, case-insensitive, and cut everything after.
_END_MARKER = re.compile(
    r"^\s*\*?\*?\*?\s*end of (this|the)? ?project gutenberg",
    re.IGNORECASE | re.MULTILINE,
)

PG19_GCS_BASE = "https://storage.googleapis.com/deepmind-gutenberg/"
PG19_MANIFEST_REPO = "deepmind/pg19"
PG19_MANIFEST_FILE = "data/train_files.txt"


class Gutenberg(Dataset):
    name = "gutenberg"
    url = PG19_GCS_BASE
    default_path = "input_gutenberg.txt"
    description = (
        "DeepMind PG-19 subset (Project Gutenberg books pre-1919). "
        "First 3000 books, ~2 GB. Set max_books=None to download all 28,602."
    )
    max_books: int | None = 300

    def prepare(self, path: str | None = None) -> str:
        path = path or self.default_path

        # if os.path.exists(path):
        #     print(f"using cached {path}")
        #     with open(path, "r", encoding="utf-8") as f:
        #         return f.read()

        manifest = hf_hub_download(
            PG19_MANIFEST_REPO, PG19_MANIFEST_FILE, repo_type="dataset"
        )
        with open(manifest) as f:
            book_paths = [line.strip() for line in f if line.strip()]
        if self.max_books is not None:
            book_paths = book_paths[: self.max_books]

        print(f"downloading {len(book_paths):,} PG-19 books to {path}")
        tmp = f"{path}.partial"
        chars = 0
        with open(tmp, "w", encoding="utf-8") as out:
            for i, rel in enumerate(book_paths):
                with urllib.request.urlopen(PG19_GCS_BASE + rel) as r:
                    raw = r.read().decode("utf-8", errors="replace")
                m = _END_MARKER.search(raw)
                if m:
                    raw = raw[: m.start()]
                block = raw.strip() + "\n\n"
                out.write(block)
                chars += len(block)
                if (i + 1) % 50 == 0 or i + 1 == len(book_paths):
                    print(
                        f"\r  {i + 1}/{len(book_paths)} books · {chars / 1e9:.2f} GB",
                        end="", flush=True,
                    )
        print()

        with open(tmp, "r", encoding="utf-8") as f:
            text = self.postprocess(f.read())
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        os.remove(tmp)
        print(f"  wrote {path} ({len(text):,} chars)")
        return text

    def postprocess(self, raw: str) -> str:
        """Drop characters that appear less often than '+' (a low-frequency
        sentinel for OCR noise / rare unicode). Filters the long-tail vocab
        without hardcoding an allowlist."""
        counts = Counter(raw)
        threshold = counts.get("+", 0)
        keep = {c for c, n in counts.items() if n > threshold}
        drop = set(counts) - keep
        if not drop:
            return raw
        print(
            f"  vocab: {len(counts)} → {len(keep)} chars "
            f"(dropped {len(drop)}: {''.join(sorted(drop))!r})"
        )
        return raw.translate(str.maketrans({c: None for c in drop}))
