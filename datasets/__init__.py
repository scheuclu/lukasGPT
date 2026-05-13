"""Text dataset registry.

Each dataset is a small subclass of `Dataset` that declares its URL,
default local path, and (optionally) a `postprocess` method to clean
or filter the raw download. Register a new one by importing its class
in this module and adding it to the list below.

Note: this is the *local* package — not the PyPI `datasets` library
from Hugging Face. We don't use that here.
"""

from .base import Dataset
from .shakespeare import Shakespeare
from .tinystories import TinyStories

_DATASETS: list[type[Dataset]] = [TinyStories, Shakespeare]
_REGISTRY: dict[str, type[Dataset]] = {cls.name: cls for cls in _DATASETS}


def get(name: str) -> Dataset:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown dataset {name!r}. Available: {names()}"
        )
    return _REGISTRY[name]()


def names() -> list[str]:
    return sorted(_REGISTRY)
