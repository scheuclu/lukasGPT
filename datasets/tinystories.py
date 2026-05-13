from .base import Dataset


class TinyStories(Dataset):
    name = "tinystories"
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    default_path = "input.txt"
    description = "Ronen Eldan's TinyStories train split (~1.9 GB of simple short stories)."

    # No postprocessing — TinyStories is already plain text, one story per
    # blank-line-delimited block. The training loop is happy with that.
