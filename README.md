
# nanogpt-lecture

Code from Karpathy's [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) lecture series, specifically the first lecture on nanoGPT. This fork extends the original with a few extras: Pydantic hyperparameter profiles, checkpoints that carry their own architecture + vocab, depth-N lookahead sampling at inference time, and a Streamlit dashboard for exploring the learned embeddings.

## Quickstart

```bash
uv sync                                       # install deps (Python ≥ 3.13)
uv run python download_checkpoints.py         # pull pretrained checkpoints (~474 MB)
uv run python gpt.py --infer checkpoints/ckpt_step_04999.pt   # generate text
```

To train your own instead (CUDA required):

```bash
uv run python gpt.py                          # writes ./checkpoints/ckpt_step_*.pt
```

At the end of training the final checkpoint is auto-uploaded to the HF Hub repo listed in `checkpoints.json` and tagged with the current git SHA (`ckpt_step_<step>_<sha>.pt`). A `dirty` suffix is appended if the working tree has uncommitted changes. Disable with `--no-upload`, or override the destination with `--upload-repo <user>/<repo>`. The script prints a JSON snippet you can paste into `checkpoints.json` to publish the upload.

## Pretrained checkpoints

The repo stays small; checkpoints live externally and are pinned by the git SHA of the training code. `checkpoints.json` is the manifest (step → URL + sha256), and `download_checkpoints.py` fetches them.

```bash
uv run python download_checkpoints.py            # download all
uv run python download_checkpoints.py --step 4999  # download just one
uv run python download_checkpoints.py --list       # show what's available
```

Downloads verify sha256 and skip files that are already present.

## Training

`gpt.py` trains a character-level transformer on `input.txt` (auto-downloaded). Architecture, training schedule, and dropout are bundled into named `Hyperparameters` profiles at the top of `gpt.py`:

- `default` — n_embd 384, 6 layers, 6 heads, block_size 512, 5000 iters
- `tiny` — small enough to iterate on CPU-class setups
- `large` — 12 layers, 12 heads, block_size 1024

Switch profiles by editing `ACTIVE_PROFILE` in `gpt.py`. Each checkpoint embeds the architecture and vocab so it's self-contained for inference.

## Inference

```bash
uv run python gpt.py --infer checkpoints/ckpt_step_04999.pt --max-new-tokens 500
```

Add lookahead sampling to sample by joint probability over a depth-N beam tree:

```bash
uv run python gpt.py --infer <ckpt> --lookahead-depth 3 --lookahead-width 4
```

## Visualizing learned embeddings

```bash
uv run streamlit run viz_embeddings.py
```

Opens an interactive dashboard with:

- 3D / 2D PCA scatter of the token embedding table (colored by character category)
- Cosine-similarity heatmap between every pair of token embeddings
- 3D / 2D PCA of the position embedding table (colored by position index)

The sidebar lets you scrub across training-step checkpoints and watch the embeddings organize over time.

## Layout

- `gpt.py` — the transformer, training loop, inference, and lookahead sampling
- `bigram.py` — the tiny bigram baseline from earlier in the lecture
- `viz_embeddings.py` — Streamlit embedding viewer
- `checkpoints.json` + `download_checkpoints.py` — manifest and downloader for published checkpoints
- `checkpoints/` — training checkpoints (gitignored; populated by training or by the downloader)

## Publishing new checkpoints (maintainers)

Checkpoints are hosted on a Hugging Face Hub model repo. `huggingface_hub` is already a regular dep, so its `hf` CLI is available via `uv run`. To publish a fresh batch:

```bash
# one-time: authenticate
uv run hf auth login

# create the model repo (one-time)
uv run hf repo create ng-video-lecture-checkpoints --type model

# copy & rename with the current git SHA so files are pinned to the training code
GIT_SHA=$(git rev-parse --short HEAD)
mkdir -p upload
for step in 0 100 500 1000 2000 4999; do
  cp checkpoints/ckpt_step_$(printf "%05d" $step).pt \
     upload/ckpt_step_$(printf "%05d" $step)_${GIT_SHA}.pt
done

# upload
uv run hf upload scheuclu/ng-video-lecture-checkpoints ./upload .

# then edit checkpoints.json: bump filenames + sha256s for each step
```

## Notes from Karpathy's original README

Sadly the video lecture did not go too deep into model initialization, which is quite important for good performance. The code in this repo will train fine, but its convergence is slower because it starts off in a not-great spot in the weight space. See [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)'s `_init_weights` for the canonical version.

## License

MIT
