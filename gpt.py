import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, model_validator
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter

import datasets as corpora
import tokenizers as tok
from checkpoint_io import git_sha, manifest_repo, sha256_file, upload_checkpoint
from tokenizers.base import Tokenizer


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
    eval_interval: int = 25
    eval_iters: int = 200
    # LR schedule: linear warmup from 0 to learning_rate over warmup_iters,
    # then ReduceLROnPlateau — multiply lr by lr_factor whenever val loss
    # hasn't improved by lr_threshold for lr_patience eval intervals.
    # Bottoms out at min_lr.
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    lr_factor: float = 0.5
    lr_patience: int = 3
    lr_threshold: float = 1e-3

    @model_validator(mode="after")
    def _check(self) -> "Hyperparameters":
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        return self

    def warmup_lr(self, iter_num: int) -> float | None:
        """LR during warmup, or None once warmup is over and the
        plateau scheduler takes over."""
        if iter_num < self.warmup_iters:
            return self.learning_rate * (iter_num + 1) / (self.warmup_iters + 1)
        return None

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
        batch_size=64,
        max_iters=5000,
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

    def __init__(self, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
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
        # Build the causal mask on the fly instead of storing it as a buffer:
        # checkpoints stay slim and we work for any T <= block_size.
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        wei = wei.masked_fill(~mask, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        head_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, dropout) for _ in range(num_heads)]
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

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, dropout)
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
            *[
                Block(hp.n_embd, hp.n_head, hp.dropout)
                for _ in range(hp.n_layer)
            ]
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


def load_model_from_checkpoint(
    path: str, device: str = "cpu"
) -> tuple[GPTLanguageModel, Tokenizer, Hyperparameters]:
    """Load a checkpoint and return (model, tokenizer, hp). Used by both the
    inference CLI in __main__ and external tools like viz_embeddings.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "hparams" not in ckpt:
        raise ValueError(f"checkpoint {path} missing 'hparams'")
    # New format: tokenizer_type + tokenizer_state. Legacy format: just `chars`.
    if "tokenizer_type" in ckpt and "tokenizer_state" in ckpt:
        tokenizer = tok.get(ckpt["tokenizer_type"])
        tokenizer.load_state_dict(ckpt["tokenizer_state"])
    elif "chars" in ckpt:
        tokenizer = tok.get("char")
        tokenizer.load_state_dict({"chars": ckpt["chars"]})
    else:
        raise ValueError(f"checkpoint {path} missing tokenizer info ('chars' or 'tokenizer_state')")

    hp = Hyperparameters(**ckpt["hparams"])
    model = GPTLanguageModel(hp, vocab_size=tokenizer.vocab_size).to(device)
    # Older checkpoints (pre-tril-removal) carry per-head 'tril' buffers we
    # no longer hold; drop them so load_state_dict doesn't complain.
    state = {k: v for k, v in ckpt["model"].items() if not k.endswith(".tril")}
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, hp


# Module-level worker state. multiprocessing.Pool's initializer runs once per
# worker process and stashes the tokenizer here so each task call doesn't
# need to (re)pickle it.
_worker_tokenizer: Tokenizer | None = None


def _init_worker(tok_type: str, tok_state: dict[str, Any]) -> None:
    global _worker_tokenizer
    _worker_tokenizer = tok.get(tok_type)
    _worker_tokenizer.load_state_dict(tok_state)


def _encode_one_chunk(chunk: str) -> list[int]:
    assert _worker_tokenizer is not None
    return _worker_tokenizer.encode(chunk)


def _encode_corpus_with_progress(
    text: str, tokenizer: Tokenizer, n_chunks: int = 200, n_workers: int | None = None
) -> torch.Tensor:
    """Encode `text` through `tokenizer` in roughly-equal-sized chunks, in
    parallel across processes. Functionally equivalent to
    `tokenizer.encode(text)` except for at most ~n_chunks tokens of slop
    at chunk boundaries — negligible against a corpus of millions of
    tokens, and the alternative is staring at a frozen prompt for hours
    on the BPE path.
    """
    chunk_size = max(1, len(text) // n_chunks)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    if n_workers is None:
        n_workers = min(len(chunks), mp.cpu_count())

    encoded: list[int] = []
    total_mb = len(text) / 1e6
    t0 = time.time()
    report_every = max(1, len(chunks) // 50)
    print(f"  encoding {len(chunks)} chunks across {n_workers} workers")
    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(tokenizer.name, tokenizer.state_dict()),
    ) as pool:
        # imap preserves input order, so concatenation gives us back the
        # original text's token sequence (modulo cross-chunk slop).
        for i, result in enumerate(pool.imap(_encode_one_chunk, chunks)):
            encoded.extend(result)
            if (i + 1) % report_every == 0 or i + 1 == len(chunks):
                done = (i + 1) / len(chunks)
                elapsed = time.time() - t0
                eta = elapsed * (1 - done) / done if done > 0 else 0.0
                print(
                    f"\r  encoding: {done * 100:5.1f}% · "
                    f"{done * total_mb:5.1f}/{total_mb:.1f} MB · "
                    f"{elapsed:4.0f}s elapsed · ETA {eta:4.0f}s",
                    end="", flush=True,
                )
    print()
    return torch.tensor(encoded, dtype=torch.long)


def _main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "checkpoints"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file. If set, skip training and just generate text.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=500, help="Number of tokens to generate."
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
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip auto-upload of the final checkpoint to HF Hub at end of training.",
    )
    parser.add_argument(
        "--upload-repo",
        default=None,
        help="HF Hub model repo to upload to. Default: `repo` field in checkpoints.json.",
    )
    parser.add_argument(
        "--dataset",
        default="gutenberg",
        choices=corpora.names(),
        help="Training corpus. See `datasets/` for available options.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Skip TensorBoard logging. By default each training run is logged under ./runs/.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Skip generating sample text at every eval step. Useful on the `large` profile where generation is slow.",
    )
    parser.add_argument(
        "--tokenizer",
        default="char",
        choices=tok.names(),
        help="Tokenizer to use. 'char' = one token per character; 'bpe' = byte-level BPE trained on the corpus.",
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=1024,
        help="Target vocab size when --tokenizer is 'bpe'. Ignored otherwise.",
    )
    args = parser.parse_args()

    torch.manual_seed(1337)

    if args.infer is not None:
        print(f"loading checkpoint from {args.infer}")
        model, tokenizer, hp = load_model_from_checkpoint(args.infer, device=device)
        print(
            f"  architecture: n_embd={hp.n_embd} n_head={hp.n_head} n_layer={hp.n_layer} "
            f"block_size={hp.block_size} dropout={hp.dropout}"
        )
        print(f"  tokenizer: {tokenizer.name} (vocab={tokenizer.vocab_size})")
        print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        if args.lookahead_depth > 1:
            print(
                f"  lookahead sampling: depth={args.lookahead_depth} width={args.lookahead_width}"
            )
            out = model.generate_lookahead(
                context,
                max_new_tokens=args.max_new_tokens,
                depth=args.lookahead_depth,
                width=args.lookahead_width,
            )
        else:
            out = model.generate(context, max_new_tokens=args.max_new_tokens)
        print(tokenizer.decode(out[0].tolist()))
        return

    assert device == "cuda", "training requires CUDA"
    hp = PROFILES[ACTIVE_PROFILE]
    print(f"Using profile '{ACTIVE_PROFILE}': {hp.model_dump()}")

    dataset = corpora.get(args.dataset)
    print(f"dataset: {args.dataset} ({dataset.description})")
    text = dataset.prepare()
    print(f"text is in RAM ({len(text):,} chars)")

    tokenizer_kwargs = (
        {"vocab_size": args.tokenizer_vocab_size} if args.tokenizer == "bpe" else {}
    )
    tokenizer = tok.get(args.tokenizer, **tokenizer_kwargs)
    print(f"tokenizer: {tokenizer.name} (training …)")
    tokenizer.train(text)
    print(f"  vocab_size={tokenizer.vocab_size}")
    vocab_size = tokenizer.vocab_size

    # Encoding the full corpus through naive BPE is slow (tens of minutes
    # on Gutenberg), so we cache the result to disk keyed on dataset +
    # tokenizer config. The cached file also stores the tokenizer state
    # to detect staleness.
    cache_id = (
        f"{tokenizer.name}_v{tokenizer.vocab_size}"
        if tokenizer.name != "char"
        else "char"
    )
    cache_path = f"{dataset.default_path}.{cache_id}.cache.pt"
    cached_state = tokenizer.state_dict()
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        if cache.get("tokenizer_state") == cached_state:
            print(f"using cached encoded corpus at {cache_path}")
            data = cache["data"]
        else:
            print(f"  cache at {cache_path} is stale, re-encoding")
            data = _encode_corpus_with_progress(text, tokenizer)
            torch.save({"tokenizer_state": cached_state, "data": data}, cache_path)
    else:
        print(f"encoding corpus ({len(text):,} chars) — this can take a while for bpe")
        data = _encode_corpus_with_progress(text, tokenizer)
        torch.save({"tokenizer_state": cached_state, "data": data}, cache_path)
        print(f"  cached encoded corpus to {cache_path}")
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

    # Initialize lm_head.bias to log unigram frequencies. With this, the
    # model's zero-context prediction starts at the corpus unigram
    # distribution rather than uniform, dropping initial loss from
    # log(vocab_size) to the unigram entropy.
    token_counts = torch.bincount(data, minlength=tokenizer.vocab_size)
    with torch.no_grad():
        freqs = token_counts.float().to(device).clamp(min=1.0)
        model.lm_head.bias.copy_((freqs / freqs.sum()).log())

    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate)
    # Plateau scheduler: drop lr only when val loss stops improving. We
    # also do a manual linear warmup over the first `warmup_iters` steps,
    # before letting this scheduler take over.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=hp.lr_factor,
        patience=hp.lr_patience,
        threshold=hp.lr_threshold,
        min_lr=hp.min_lr,
    )

    writer: SummaryWriter | None = None
    if not args.no_tensorboard:
        run_name = f"{datetime.now():%Y%m%d-%H%M%S}_{ACTIVE_PROFILE}_{args.dataset}"
        log_dir = os.path.join("runs", run_name)
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text("hparams", f"```\n{json.dumps(hp.model_dump(), indent=2)}\n```")
        writer.add_text("profile", ACTIVE_PROFILE)
        writer.add_text("dataset", args.dataset)
        writer.add_text("tokenizer", f"{tokenizer.name} (vocab={tokenizer.vocab_size})")
        n_total = int(token_counts.sum().item())
        # pyright loses the int type through argsort().tolist(); the cast is safe.
        sorted_ids: list[int] = token_counts.argsort(descending=True).tolist()  # pyright: ignore[reportUnknownVariableType]
        rows = ["| rank | id | token | count | freq |", "|---|---|---|---|---|"]
        for rank, token_id in enumerate(sorted_ids):
            count = int(token_counts[token_id].item())
            if count == 0:
                break
            tok_str = tokenizer.decode([token_id])
            rows.append(
                f"| {rank} | {token_id} | `{tok_str!r}` | {count:,} | {100 * count / n_total:.3f}% |"
            )
        writer.add_text("tokens", "\n".join(rows))
        writer.add_histogram("token_distribution", data, 0)
        print(f"tensorboard: logging to {log_dir}")

    last_saved_step: int | None = None
    interrupted = False
    try:
        for iter in range(hp.max_iters):
            # During warmup, overwrite the optimizer's lr manually. After
            # warmup, the scheduler owns it (it modifies param_groups in
            # place when it decides to step down).
            warmup = hp.warmup_lr(iter)
            if warmup is not None:
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup

            if iter % hp.eval_interval == 0 or iter == hp.max_iters - 1:
                losses = estimate_loss()
                val_loss = losses["val"].item()
                lr_before = optimizer.param_groups[0]["lr"]
                # Let the plateau scheduler decide whether to drop lr,
                # but not during warmup — its internal "best so far"
                # tracking shouldn't see the early ramp-up noise.
                if iter >= hp.warmup_iters:
                    scheduler.step(val_loss)
                lr_now = optimizer.param_groups[0]["lr"]
                if lr_now < lr_before:
                    print(
                        f"  lr reduced: {lr_before:.2e} → {lr_now:.2e} (val loss plateau)"
                    )
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}, lr {lr_now:.2e}"
                )
                if writer is not None:
                    writer.add_scalar("loss/train", losses["train"].item(), iter)
                    writer.add_scalar("loss/val", val_loss, iter)
                    writer.add_scalar("lr", lr_now, iter)
                    if not args.no_sample:
                        model.eval()
                        with torch.no_grad():
                            ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
                            sample = model.generate(ctx, max_new_tokens=200)
                        model.train()
                        writer.add_text(
                            "sample",
                            f"```\n{tokenizer.decode(sample[0].tolist())}\n```",
                            iter,
                        )
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"ckpt_{ACTIVE_PROFILE}_step_{iter:05d}.pt",
                )
                ckpt_payload: dict[str, object] = {
                    "iter": iter,
                    "profile": ACTIVE_PROFILE,
                    "model": model.state_dict(),
                    "train_loss": losses["train"].item(),
                    "val_loss": losses["val"].item(),
                    "tokenizer_type": tokenizer.name,
                    "tokenizer_state": tokenizer.state_dict(),
                    "hparams": hp.architecture_dict(),
                }
                # Keep `chars` for backwards compat with viz_embeddings /
                # export_onnx / the published checkpoints — only meaningful for
                # the char tokenizer.
                if isinstance(tokenizer, tok.CharTokenizer):
                    ckpt_payload["chars"] = tokenizer.chars
                torch.save(ckpt_payload, ckpt_path)
                last_saved_step = iter
                print(f"  saved checkpoint to {ckpt_path}")

            xb, yb = get_batch("train")
            logits, loss = model(xb, yb)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    except KeyboardInterrupt:
        interrupted = True
        print()
        print("training interrupted by user (Ctrl-C)")
        if last_saved_step is None:
            print("  no checkpoint saved yet; nothing to upload")
        else:
            print(f"  last saved checkpoint: step {last_saved_step}")

    if writer is not None:
        writer.close()

    # Only run the final-generation sample on clean completion; on Ctrl-C the
    # user wants to get out, not wait for 500 tokens of inference.
    if not interrupted:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(
            tokenizer.decode(
                model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()
            )
        )

    if args.no_upload or last_saved_step is None:
        return

    repo = args.upload_repo or manifest_repo()
    sha = git_sha()
    final_step = last_saved_step
    final_ckpt = os.path.join(
        checkpoint_dir,
        f"ckpt_{ACTIVE_PROFILE}_step_{final_step:05d}.pt",
    )
    if not repo:
        print(
            "upload: skipped (no repo configured; use --upload-repo or set `repo` in checkpoints.json)"
        )
        return
    if not sha:
        print("upload: skipped (not in a git repo)")
        return
    if not os.path.exists(final_ckpt):
        print(f"upload: skipped ({final_ckpt} not found)")
        return

    print(
        f"upload: pushing {final_ckpt} to {repo} (profile={ACTIVE_PROFILE}, sha={sha})"
    )
    try:
        remote = upload_checkpoint(final_ckpt, ACTIVE_PROFILE, final_step, sha, repo)
    except Exception as e:
        fallback = f"ckpt_{ACTIVE_PROFILE}_step_{final_step:05d}_{sha}.pt"
        print(f"upload: FAILED ({type(e).__name__}: {e})")
        print(f"  rerun later with: uv run hf upload {repo} {final_ckpt} {fallback}")
        return
    sha256 = sha256_file(final_ckpt)
    print(f"upload: ok → {repo}/{remote}")
    print()
    print("Manifest snippet — paste into checkpoints.json under `checkpoints`:")
    print(
        json.dumps(
            {
                "profile": ACTIVE_PROFILE,
                "step": final_step,
                "git_sha": sha,
                "filename": remote,
                "sha256": sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
