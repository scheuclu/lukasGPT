"""Base class for text training corpora.

A `Dataset` knows its URL, default local cache path, and how to
postprocess the raw download. `prepare()` is the entry point: it
downloads (if not cached), runs `postprocess`, writes the result to
disk, and returns the final text.
"""

import os
import urllib.request


class Dataset:
    name: str = ""
    url: str = ""
    default_path: str = ""
    description: str = ""

    def prepare(self, path: str | None = None) -> str:
        """Return the processed training text, downloading if needed.

        Writes the processed text to `path` (or `self.default_path`)
        and uses that as a cache on subsequent calls.
        """
        path = path or self.default_path
        if not path:
            raise ValueError(
                f"dataset {self.name!r} has no default_path; pass one explicitly"
            )

        if not os.path.exists(path):
            tmp = f"{path}.partial"
            print(f"downloading {self.name} from {self.url}")
            urllib.request.urlretrieve(self.url, tmp, reporthook=_progress(path))
            print()
            with open(tmp, "r", encoding="utf-8") as f:
                raw = f.read()
            os.remove(tmp)
            text = self.postprocess(raw)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  wrote {path} ({len(text):,} chars)")
        else:
            print(f"using cached {path}")

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def postprocess(self, raw: str) -> str:
        """Transform raw downloaded text. Default: pass through unchanged.

        Override in subclasses for stripping markup, filtering by
        language, joining JSONL records, etc.
        """
        return raw


def _progress(label: str):
    def hook(block_num: int, block_size: int, total_size: int) -> None:
        done = block_num * block_size
        if total_size > 0:
            pct = min(100, done * 100 // total_size)
            print(
                f"\r  {label}: {done / 1e9:.2f} / {total_size / 1e9:.2f} GB ({pct}%)",
                end="", flush=True,
            )
    return hook
