from .base import Dataset


class Shakespeare(Dataset):
    name = "shakespeare"
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    default_path = "input_shakespeare.txt"
    description = "Karpathy's tiny-shakespeare corpus (~1 MB)."
