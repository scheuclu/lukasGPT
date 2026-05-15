"""Byte-level BPE tokenizer.

Starts with the 256 byte tokens and greedily learns merges of the
most-frequent adjacent pair until the target vocab size is reached.
Same algorithm as Karpathy's minbpe; inlined here so the repo stays
self-contained.

Pros vs char-level: ~3-4x sequence compression on English, so the
same block_size covers more actual context.
Cons: bigger embedding + lm_head. Training is still pure-Python (one-time
cost on a small corpus sample). Encoding is JIT-compiled via numba —
the hot path is a tight loop over int64 arrays with O(1) merge lookup
against a precomputed (vocab_size, vocab_size) table.
"""

from collections import Counter
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit  # pyright: ignore[reportUnknownVariableType]

from .base import Tokenizer


def _pair_counts(ids: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(ids, ids[1:]))


def _merge_py(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


@njit(cache=True)
def _bpe_encode_jit(
    ids: npt.NDArray[np.int64], merge_lookup: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Greedy BPE encode. At each iteration finds the lowest-new_id (=
    earliest-learned, highest priority) merge currently present in `ids`
    and applies it to all of its occurrences. Loops until no applicable
    merge remains. `merge_lookup[a, b]` is the new_id for pair (a, b) or
    -1 if no merge exists.
    """
    n = len(ids)
    if n < 2:
        return ids

    while True:
        found = False
        best_a = np.int64(0)
        best_b = np.int64(0)
        best_new_id = np.int64(0)
        for i in range(n - 1):
            a = ids[i]
            b = ids[i + 1]
            nid = merge_lookup[a, b]
            if nid >= 0 and (not found or nid < best_new_id):
                found = True
                best_a = a
                best_b = b
                best_new_id = nid
        if not found:
            break

        out = np.empty(n, dtype=np.int64)
        out_idx = 0
        i = 0
        while i < n:
            if i + 1 < n and ids[i] == best_a and ids[i + 1] == best_b:
                out[out_idx] = best_new_id
                out_idx += 1
                i += 2
            else:
                out[out_idx] = ids[i]
                out_idx += 1
                i += 1
        ids = out[:out_idx]
        n = out_idx

    return ids


class BPETokenizer(Tokenizer):
    name = "bpe"

    def __init__(self, vocab_size: int = 1024) -> None:
        # `vocab_size` is the *target*; train() runs at most
        # (vocab_size - 256) merges.
        self._target = vocab_size
        self._merges: list[tuple[tuple[int, int], int]] = []
        self._merge_index: dict[tuple[int, int], int] = {}
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # Lazy: a (vocab_size, vocab_size) int64 array of merge -> new_id
        # (or -1), built once after train/load for O(1) lookup in the JIT
        # encode loop.
        self._merge_lookup: npt.NDArray[np.int64] | None = None

    def _build_merge_lookup(self) -> None:
        size = max(self._target, 256)
        lookup = np.full((size, size), -1, dtype=np.int64)
        for (a, b), n in self._merges:
            lookup[a, b] = n
        self._merge_lookup = lookup

    def train(self, corpus: str, sample_chars: int = 5_000_000) -> None:
        """Learn BPE merges from `corpus`.

        For corpora longer than `sample_chars`, train merges on a prefix
        of that length — full-corpus training under naive Python takes
        prohibitively long, and the prefix typically converges to
        nearly-identical merges.
        """
        text = corpus[:sample_chars] if sample_chars else corpus
        ids = list(text.encode("utf-8"))
        n_merges = max(0, self._target - 256)
        for i in range(n_merges):
            stats = _pair_counts(ids)
            if not stats:
                break
            top_pair = max(stats.items(), key=lambda kv: kv[1])[0]
            new_id = 256 + i
            ids = _merge_py(ids, top_pair, new_id)
            self._merges.append((top_pair, new_id))
            self._merge_index[top_pair] = new_id
            self._vocab[new_id] = self._vocab[top_pair[0]] + self._vocab[top_pair[1]]
        self._build_merge_lookup()

    def encode(self, text: str) -> list[int]:
        if self._merge_lookup is None:
            self._build_merge_lookup()
        assert self._merge_lookup is not None
        # bytes() → uint8 view → writable int64 array for numba
        ids_arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        result = _bpe_encode_jit(ids_arr, self._merge_lookup)
        return result.tolist()

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._vocab[i] for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def state_dict(self) -> dict[str, Any]:
        return {
            "merges": [(list(p), n) for p, n in self._merges],
            "target_vocab_size": self._target,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._target = state["target_vocab_size"]
        self._merges = [((p[0], p[1]), n) for p, n in state["merges"]]
        self._merge_index = {p: n for p, n in self._merges}
        self._vocab = {i: bytes([i]) for i in range(256)}
        for (p1, p2), new_id in self._merges:
            self._vocab[new_id] = self._vocab[p1] + self._vocab[p2]
        # Invalidate; the next encode() call will rebuild it.
        self._merge_lookup = None
