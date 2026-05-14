"""Base tokenizer class.

A `Tokenizer` maps text to a sequence of integer token IDs and back.
It can optionally be `train`ed on a corpus to learn its vocabulary,
and round-trips through a torch checkpoint via state_dict / load_state_dict.
"""

from abc import ABC, abstractmethod
from typing import Any


class Tokenizer(ABC):
    """Maps text ↔ integer token IDs."""

    name: str = ""

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    def train(self, corpus: str) -> None:
        """Fit the tokenizer to a text corpus. Default: no-op."""
        return None

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
