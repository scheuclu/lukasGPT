"""Tokenizer registry.

Maps text ↔ integer token IDs. Each tokenizer is a small subclass of
`Tokenizer`. Register a new one by importing its class in this module
and adding it to the list below.

Note: this is the *local* package — not the PyPI `tokenizers` library
from Hugging Face. We don't use that here.
"""

from typing import Any

from .base import Tokenizer
from .bpe import BPETokenizer
from .char import CharTokenizer

_TOKENIZERS: list[type[Tokenizer]] = [CharTokenizer, BPETokenizer]
_REGISTRY: dict[str, type[Tokenizer]] = {cls.name: cls for cls in _TOKENIZERS}


def get(name: str, **kwargs: Any) -> Tokenizer:
    if name not in _REGISTRY:
        raise KeyError(f"unknown tokenizer {name!r}. Available: {names()}")
    return _REGISTRY[name](**kwargs)


def names() -> list[str]:
    return sorted(_REGISTRY)
