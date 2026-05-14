"""Byte-level BPE tokenizer.

Starts with the 256 byte tokens and greedily learns merges of the
most-frequent adjacent pair until the target vocab size is reached.
Same algorithm as Karpathy's minbpe; inlined here so the repo stays
self-contained.

Pros vs char-level: ~3-4x sequence compression on English, so the
same block_size covers more actual context.
Cons: bigger embedding + lm_head, and the pure-Python implementation
is slow — both training and encoding scale as O(seq * merges). Use
`sample_chars` to train merges on a prefix; the full corpus is then
encoded once and cached to disk via `gpt.py`.
"""

from collections import Counter
from typing import Any

from .base import Tokenizer


def _pair_counts(ids: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(ids, ids[1:]))


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
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


class BPETokenizer(Tokenizer):
    name = "bpe"

    def __init__(self, vocab_size: int = 1024) -> None:
        # `vocab_size` is the *target*; train() runs at most
        # (vocab_size - 256) merges.
        self._target = vocab_size
        self._merges: list[tuple[tuple[int, int], int]] = []
        self._merge_index: dict[tuple[int, int], int] = {}
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, corpus: str, sample_chars: int = 5_000_000) -> None:
        """Learn BPE merges from `corpus`.

        For corpora longer than `sample_chars`, train merges on a
        prefix of that length — full-corpus training under naive
        Python takes prohibitively long, and the prefix typically
        converges to nearly-identical merges.
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
            ids = _merge(ids, top_pair, new_id)
            self._merges.append((top_pair, new_id))
            self._merge_index[top_pair] = new_id
            self._vocab[new_id] = self._vocab[top_pair[0]] + self._vocab[top_pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _pair_counts(ids)
            # Apply the merge that was learned earliest among any pair
            # currently present (= the highest-priority merge).
            best: tuple[int, int] | None = None
            best_rank = float("inf")
            for pair in stats:
                rank = self._merge_index.get(pair)
                if rank is not None and rank < best_rank:
                    best = pair
                    best_rank = rank
            if best is None:
                break
            ids = _merge(ids, best, self._merge_index[best])
        return ids

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
