import argparse
import os

import torch
import torch.nn as nn
from pydantic import BaseModel, model_validator
from torch.nn import functional as F


class Hyperparameters(BaseModel):
    # Architecture (persisted to checkpoint, restored on inference)
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 512
    dropout: float = 0.2
    # Training (not persisted; only used in the training branch)
    batch_size: int = 128
    max_iters: int = 5000
    eval_interval: int = 50
    eval_iters: int = 200
    learning_rate: float = 3e-4

    @model_validator(mode="after")
    def _check(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        return self

    def architecture_dict(self) -> dict:
        return {
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "block_size": self.block_size,
            "dropout": self.dropout,
        }


PROFILES: dict[str, Hyperparameters] = {
    "default": Hyperparameters(),
    "tiny": Hyperparameters(
        n_embd=64,
        n_head=4,
        n_layer=2,
        block_size=64,
        batch_size=32,
        max_iters=1000,
    ),
    "large": Hyperparameters(
        n_embd=768,
        n_head=12,
        n_layer=12,
        block_size=1024,
        batch_size=64,
        max_iters=20000,
    ),
}

ACTIVE_PROFILE = "default"

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda"
checkpoint_dir = "checkpoints"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--infer",
    type=str,
    default=None,
    help="Path to a checkpoint .pt file. If set, skip training and just generate text.",
)
parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=500,
    help="Number of tokens to generate.",
)
parser.add_argument(
    "--lookahead-depth",
    type=int,
    default=1,
    help="If >1, use lookahead sampling: expand a depth-N tree and sample a path by joint probability.",
)
parser.add_argument(
    "--lookahead-width",
    type=int,
    default=4,
    help="Branching factor at each level of the lookahead tree (top-K candidates).",
)
args = parser.parse_args()

torch.manual_seed(1337)


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    @torch.no_grad()
    def generate_lookahead(self, idx, max_new_tokens, depth=2, width=4):
        # At each step, expand a depth-`depth` tree with branching factor
        # `width`, then sample one full path proportional to its joint
        # probability and commit all of its tokens at once. Top-1-greedy at
        # each token can miss high-joint-probability sequences; the tree
        # lets a low-prob first token survive if its continuation is strong.
        assert idx.size(0) == 1, "lookahead generate only supports batch size 1"
        remaining = max_new_tokens
        while remaining > 0:
            commit = min(depth, remaining)
            leaves = idx  # (n_leaves, T), starts as (1, T)
            log_probs = torch.zeros(1, device=idx.device)  # joint log-prob per leaf
            for _ in range(commit):
                logits, _ = self(leaves[:, -block_size:])
                step_logp = F.log_softmax(logits[:, -1, :], dim=-1)  # (n_leaves, V)
                k = min(width, step_logp.size(-1))
                top_logp, top_idx = step_logp.topk(k, dim=-1)  # (n_leaves, k)
                leaves = torch.cat(
                    [leaves.repeat_interleave(k, dim=0), top_idx.reshape(-1, 1)],
                    dim=1,
                )
                log_probs = log_probs.repeat_interleave(k) + top_logp.reshape(-1)
            # softmax over joint log-probs => sample a leaf in proportion to its
            # joint probability (relative to the other leaves in this tree)
            path_probs = F.softmax(log_probs, dim=-1)
            chosen = torch.multinomial(path_probs, num_samples=1)
            idx = torch.cat([idx, leaves[chosen, -commit:]], dim=1)
            remaining -= commit
        return idx


if args.infer is not None:
    print(f"loading checkpoint from {args.infer}")
    ckpt = torch.load(args.infer, map_location=device)
    if "chars" not in ckpt:
        raise ValueError(
            f"Checkpoint {args.infer} has no 'chars' field. "
            "Retrain with the updated script to embed the vocab in the checkpoint."
        )
    if "hparams" not in ckpt:
        raise ValueError(
            f"Checkpoint {args.infer} has no 'hparams' field. "
            "Retrain with the updated script to embed the architecture hyperparameters."
        )
    chars = ckpt["chars"]
    vocab_size = len(chars)
    itos = {i: ch for i, ch in enumerate(chars)}
    decode = lambda l: "".join([itos[i] for i in l])

    hp = Hyperparameters(**ckpt["hparams"])
    n_embd = hp.n_embd
    n_head = hp.n_head
    n_layer = hp.n_layer
    block_size = hp.block_size
    dropout = hp.dropout
    print(
        f"  architecture: n_embd={n_embd} n_head={n_head} n_layer={n_layer} "
        f"block_size={block_size} dropout={dropout}"
    )

    model = GPTLanguageModel()
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")
    model.load_state_dict(ckpt["model"])
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    if args.lookahead_depth > 1:
        print(
            f"  lookahead sampling: depth={args.lookahead_depth} width={args.lookahead_width}"
        )
        out = m.generate_lookahead(
            context,
            max_new_tokens=args.max_new_tokens,
            depth=args.lookahead_depth,
            width=args.lookahead_width,
        )
    else:
        out = m.generate(context, max_new_tokens=args.max_new_tokens)
    print(decode(out[0].tolist()))
else:
    hp = PROFILES[ACTIVE_PROFILE]
    print(f"Using profile '{ACTIVE_PROFILE}': {hp.model_dump()}")
    n_embd = hp.n_embd
    n_head = hp.n_head
    n_layer = hp.n_layer
    block_size = hp.block_size
    dropout = hp.dropout
    batch_size = hp.batch_size
    max_iters = hp.max_iters
    eval_interval = hp.eval_interval
    eval_iters = hp.eval_iters
    learning_rate = hp.learning_rate

    INPUT_PATH = "input.txt"
    INPUT_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"

    if not os.path.exists(INPUT_PATH):
        import urllib.request

        print(f"{INPUT_PATH} not found. Downloading from {INPUT_URL} (~1.9 GB)...")

        def _progress(block_num, block_size_bytes, total_size):
            downloaded = block_num * block_size_bytes
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                print(
                    f"\r  {downloaded / 1e9:.2f} / {total_size / 1e9:.2f} GB ({pct}%)",
                    end="",
                    flush=True,
                )

        urllib.request.urlretrieve(INPUT_URL, INPUT_PATH, reporthook=_progress)
        print()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print("text is in RAM")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    model = GPTLanguageModel()
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets and save a checkpoint
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step_{iter:05d}.pt")
            torch.save(
                {
                    "iter": iter,
                    "model": model.state_dict(),
                    "train_loss": losses["train"].item(),
                    "val_loss": losses["val"].item(),
                    "chars": chars,
                    "hparams": hp.architecture_dict(),
                },
                ckpt_path,
            )
            print(f"  saved checkpoint to {ckpt_path}")

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()))
