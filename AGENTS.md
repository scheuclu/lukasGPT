# AGENTS.md

Guidance for AI coding agents working in this repository. Follows the [agents.md](https://agents.md) spec.

## What this is

A hand-extended fork of Karpathy's `ng-video-lecture` — a character-level transformer trained on Shakespeare-ish text. The repo is intentionally small and pedagogical; **prefer minimal, readable changes** over framework-y abstractions.

## Setup

- Python ≥ 3.13, managed with `uv`. Install deps with `uv sync`.
- `gpt.py` asserts `cuda` is available (`assert device == "cuda"`); training and inference are CUDA-only.
- There is no test suite, linter config, or CI. "Does it run?" is the bar — actually run the script you changed.

## Commands

| What | How |
|------|-----|
| Train | `uv run python gpt.py` |
| Infer | `uv run python gpt.py --infer <ckpt> --max-new-tokens 500` |
| Infer + lookahead | `uv run python gpt.py --infer <ckpt> --lookahead-depth 3 --lookahead-width 4` |
| Viz dashboard | `uv run streamlit run viz_embeddings.py` |

## Code layout

- `gpt.py` — single-file transformer (model + training loop + inference). All hyperparameters live in the `Hyperparameters` Pydantic model and named `PROFILES` dict near the top.
- `bigram.py` — tiny bigram baseline; left as-is for reference, do not refactor unless asked.
- `viz_embeddings.py` — Streamlit dashboard. Reads checkpoints directly via `torch.load`; does not import from `gpt.py`.
- `checkpoints/ckpt_step_*.pt` — self-contained: every checkpoint stores `model`, `chars`, and `hparams`.

## Conventions

- **Style**: terse, no emojis, no decorative comments. Match the existing files — if a function doesn't need a docstring, don't add one.
- **Hyperparameters**: any new tunable goes on the `Hyperparameters` Pydantic model, with profile entries updated if the new value matters. Do not scatter constants throughout the file.
- **Checkpoint format**: if you change what's saved, also update the loader at the top of the inference branch — `chars` and `hparams` are required keys.
- **Dependencies**: add via `uv add <pkg>`; both `pyproject.toml` and `uv.lock` must be committed together.

## Git & PR workflow

- **The remote `upstream` points at `karpathy/ng-video-lecture`. Never push or open PRs there.** All work goes to `origin` (`scheuclu/ng-video-lecture`).
- When creating PRs with `gh`, **always pass `--repo scheuclu/ng-video-lecture`** — otherwise `gh` defaults to the parent of the fork (karpathy's repo) and pings strangers.
- Branch naming convention from recent history: `feature/<short-kebab-name>` (see `feature/lookahead-sampling`, `feature/hparam-profiles`).
- Commits are imperative, capital first letter, period at end (e.g. `Add depth-N lookahead sampling for inference.`).
- Default to draft PRs (`gh pr create --draft`) for review.

## Things not to touch

- **`checkpoints/`** is gitignored but contains hours of training. Never `rm`, never `git clean -fdx`, never overwrite. If a checkpoint seems stale or wrong, ask before deleting.
- **`input.txt` / `input_shakespeare.txt`** are gitignored data files. Auto-downloaded if missing, but don't delete them casually — re-downloading is slow.
- **`viz/`** is gitignored; if it exists it's stale output from an earlier non-Streamlit prototype. Safe to ignore, ask before deleting.

## Common pitfalls

- `nn.Embedding` weights are trainable parameters — both `token_embedding_table` and `position_embedding_table` get updated by the optimizer. Don't add `.requires_grad = False` "to be safe."
- The model is char-level. `vocab_size` is derived from `chars` in the checkpoint (currently 243 for the latest training data, not 65 as a pure Shakespeare run would give). Always read it from the checkpoint, never hardcode.
- `block_size` is the *maximum* context window. Position embeddings have exactly `block_size` rows; feeding longer sequences will index out of range.
