import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from pydantic import BaseModel, model_validator
from torch.nn import functional as F

import datasets as corpora


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
    def _check(self) -> "Hyperparameters":
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        return self

    def architecture_dict(self) -> dict[str, int | float]:
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


class Head(nn.Module):
    """one head of self-attention"""

    # Declared so pyright knows the registered buffer is a Tensor.
    tril: torch.Tensor

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, n_embd: int, num_heads: int, head_size: int,
                 block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, hp: Hyperparameters, vocab_size: int):
        super().__init__()
        self.hp = hp
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, hp.n_embd)
        self.position_embedding_table = nn.Embedding(hp.block_size, hp.n_embd)
        self.blocks = nn.Sequential(
            *[Block(hp.n_embd, hp.n_head, hp.block_size, hp.dropout)
              for _ in range(hp.n_layer)]
        )
        self.ln_f = nn.LayerNorm(hp.n_embd)
        self.lm_head = nn.Linear(hp.n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        device = idx.device

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

    @torch.no_grad()
    def residuals(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Run a forward pass and return the residual stream at every layer.

        Returns a list of (B, T, n_embd) tensors:
          [0]            after tok+pos embedding (pre-block input)
          [1..n_layer]   after each transformer block
          [n_layer+1]    after the final layer norm
        """
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        out: list[torch.Tensor] = [x]
        for block in self.blocks:
            x = block(x)
            out.append(x)
        out.append(self.ln_f(x))
        return out

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        block_size = self.hp.block_size
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_lookahead(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        depth: int = 2,
        width: int = 4,
    ) -> torch.Tensor:
        # At each step, expand a depth-`depth` tree with branching factor
        # `width`, then sample one full path proportional to its joint
        # probability and commit all of its tokens at once.
        assert idx.size(0) == 1, "lookahead generate only supports batch size 1"
        block_size = self.hp.block_size
        remaining = max_new_tokens
        while remaining > 0:
            commit = min(depth, remaining)
            leaves = idx
            log_probs = torch.zeros(1, device=idx.device)
            for _ in range(commit):
                logits, _ = self(leaves[:, -block_size:])
                step_logp = F.log_softmax(logits[:, -1, :], dim=-1)
                k = min(width, step_logp.size(-1))
                top_logp, top_idx = step_logp.topk(k, dim=-1)
                leaves = torch.cat(
                    [leaves.repeat_interleave(k, dim=0), top_idx.reshape(-1, 1)],
                    dim=1,
                )
                log_probs = log_probs.repeat_interleave(k) + top_logp.reshape(-1)
            path_probs = F.softmax(log_probs, dim=-1)
            chosen = torch.multinomial(path_probs, num_samples=1)
            idx = torch.cat([idx, leaves[chosen, -commit:]], dim=1)
            remaining -= commit
        return idx


def _git_sha() -> str | None:
    """Short HEAD SHA with `-dirty` suffix if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None
    if subprocess.run(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL).returncode:
        sha = f"{sha}-dirty"
    return sha


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _manifest_repo() -> str | None:
    """Read the HF Hub repo from checkpoints.json if present."""
    try:
        with open("checkpoints.json") as f:
            return json.load(f).get("repo")
    except Exception:
        return None


def upload_checkpoint(local_path: str, profile: str, step: int,
                      git_sha: str, repo: str) -> str:
    """Upload `local_path` to the given HF Hub model repo, renaming the
    remote file to `ckpt_<profile>_step_<step>_<sha>.pt`. Creates the
    repo if it doesn't exist. Returns the remote filename on success."""
    from huggingface_hub import create_repo, upload_file
    remote = f"ckpt_{profile}_step_{step:05d}_{git_sha}.pt"
    create_repo(repo, repo_type="model", exist_ok=True)
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote,
        repo_id=repo,
        repo_type="model",
    )
    return remote


def load_model_from_checkpoint(path: str, device: str = "cpu") -> tuple[GPTLanguageModel, list[str], Hyperparameters]:
    """Load a checkpoint and return (model, chars, hp). Used by both the
    inference CLI in __main__ and external tools like viz_embeddings.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "chars" not in ckpt or "hparams" not in ckpt:
        raise ValueError(f"checkpoint {path} missing 'chars' or 'hparams'")
    chars = ckpt["chars"]
    hp = Hyperparameters(**ckpt["hparams"])
    model = GPTLanguageModel(hp, vocab_size=len(chars)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, chars, hp


def _main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "checkpoints"

    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str, default=None,
                        help="Path to a checkpoint .pt file. If set, skip training and just generate text.")
    parser.add_argument("--max-new-tokens", type=int, default=500,
                        help="Number of tokens to generate.")
    parser.add_argument("--lookahead-depth", type=int, default=1,
                        help="If >1, use lookahead sampling: expand a depth-N tree and sample a path by joint probability.")
    parser.add_argument("--lookahead-width", type=int, default=4,
                        help="Branching factor at each level of the lookahead tree (top-K candidates).")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip auto-upload of the final checkpoint to HF Hub at end of training.")
    parser.add_argument("--upload-repo", default=None,
                        help="HF Hub model repo to upload to. Default: `repo` field in checkpoints.json.")
    parser.add_argument("--dataset", default="tinystories", choices=corpora.names(),
                        help="Training corpus. See `datasets/` for available options.")
    args = parser.parse_args()

    torch.manual_seed(1337)

    if args.infer is not None:
        print(f"loading checkpoint from {args.infer}")
        model, chars, hp = load_model_from_checkpoint(args.infer, device=device)
        itos = {i: ch for i, ch in enumerate(chars)}
        decode: Callable[[Iterable[int]], str] = lambda l: "".join([itos[i] for i in l])
        print(
            f"  architecture: n_embd={hp.n_embd} n_head={hp.n_head} n_layer={hp.n_layer} "
            f"block_size={hp.block_size} dropout={hp.dropout}"
        )
        print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        if args.lookahead_depth > 1:
            print(f"  lookahead sampling: depth={args.lookahead_depth} width={args.lookahead_width}")
            out = model.generate_lookahead(
                context,
                max_new_tokens=args.max_new_tokens,
                depth=args.lookahead_depth,
                width=args.lookahead_width,
            )
        else:
            out = model.generate(context, max_new_tokens=args.max_new_tokens)
        print(decode(out[0].tolist()))
        return

    assert device == "cuda", "training requires CUDA"
    hp = PROFILES[ACTIVE_PROFILE]
    print(f"Using profile '{ACTIVE_PROFILE}': {hp.model_dump()}")

    dataset = corpora.get(args.dataset)
    print(f"dataset: {args.dataset} ({dataset.description})")
    text = dataset.prepare()
    print(f"text is in RAM ({len(text):,} chars)")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode: Callable[[str], list[int]] = lambda s: [stoi[c] for c in s]
    decode: Callable[[Iterable[int]], str] = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - hp.block_size, (hp.batch_size,))
        x = torch.stack([d[i : i + hp.block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + hp.block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss() -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(hp.eval_iters)
            for k in range(hp.eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                assert loss is not None
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    model = GPTLanguageModel(hp, vocab_size).to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate)

    for iter in range(hp.max_iters):
        if iter % hp.eval_interval == 0 or iter == hp.max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"ckpt_{ACTIVE_PROFILE}_step_{iter:05d}.pt",
            )
            torch.save(
                {
                    "iter": iter,
                    "profile": ACTIVE_PROFILE,
                    "model": model.state_dict(),
                    "train_loss": losses["train"].item(),
                    "val_loss": losses["val"].item(),
                    "chars": chars,
                    "hparams": hp.architecture_dict(),
                },
                ckpt_path,
            )
            print(f"  saved checkpoint to {ckpt_path}")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()))

    if args.no_upload:
        return

    repo = args.upload_repo or _manifest_repo()
    sha = _git_sha()
    final_step = hp.max_iters - 1
    final_ckpt = os.path.join(
        checkpoint_dir,
        f"ckpt_{ACTIVE_PROFILE}_step_{final_step:05d}.pt",
    )
    if not repo:
        print("upload: skipped (no repo configured; use --upload-repo or set `repo` in checkpoints.json)")
        return
    if not sha:
        print("upload: skipped (not in a git repo)")
        return
    if not os.path.exists(final_ckpt):
        print(f"upload: skipped ({final_ckpt} not found)")
        return

    print(f"upload: pushing {final_ckpt} to {repo} (profile={ACTIVE_PROFILE}, sha={sha})")
    try:
        remote = upload_checkpoint(final_ckpt, ACTIVE_PROFILE, final_step, sha, repo)
    except Exception as e:
        fallback = f"ckpt_{ACTIVE_PROFILE}_step_{final_step:05d}_{sha}.pt"
        print(f"upload: FAILED ({type(e).__name__}: {e})")
        print(f"  rerun later with: uv run hf upload {repo} {final_ckpt} {fallback}")
        return
    sha256 = _sha256_file(final_ckpt)
    print(f"upload: ok → {repo}/{remote}")
    print()
    print("Manifest snippet — paste into checkpoints.json under `checkpoints`:")
    print(json.dumps({
        "profile": ACTIVE_PROFILE,
        "step": final_step,
        "git_sha": sha,
        "filename": remote,
        "sha256": sha256,
    }, indent=2))


if __name__ == "__main__":
    _main()
