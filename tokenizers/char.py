"""Character-level tokenizer.

Each unique character in the training corpus becomes a token. Tiny
vocabularies (~85-250 for English-ish text), no OOV concept, no
sequence compression — every input character is one token.
"""

from typing import Any

from .base import Tokenizer


class CharTokenizer(Tokenizer):
    name = "char"

    def __init__(self) -> None:
        self._chars: list[str] = []
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}

    def train(self, corpus: str) -> None:
        self._chars = sorted(set(corpus))
        self._rebuild_maps()

    def encode(self, text: str) -> list[int]:
        # Silently skip OOV characters (matches the prior inline behavior).
        return [self._stoi[c] for c in text if c in self._stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    @property
    def chars(self) -> list[str]:
        return self._chars

    def state_dict(self) -> dict[str, Any]:
        return {"chars": list(self._chars)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._chars = list(state["chars"])
        self._rebuild_maps()

    def _rebuild_maps(self) -> None:
        self._stoi = {c: i for i, c in enumerate(self._chars)}
        self._itos = {i: c for i, c in enumerate(self._chars)}
