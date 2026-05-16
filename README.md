
# lukasGPT

[try here](https://scheuclu.github.io/lukasGPT)

Started as a fork of Karpathy's [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) lecture series, specifically the first lecture on nanoGPT. This fork extends the original with a few extras: Pydantic hyperparameter profiles, checkpoints that carry their own architecture + vocab, depth-N lookahead sampling at inference time, and a Streamlit dashboard for exploring the learned embeddings.

## Quickstart

```bash
uv sync                                                    # install deps (Python ≥ 3.13)
uv run python download_checkpoints.py                      # pull pretrained checkpoints (~474 MB)
uv run python gpt.py --infer checkpoints/ckpt_default_step_04999.pt   # generate text
```

To train your own instead (CUDA required):

```bash
uv run python gpt.py                              # default dataset: gutenberg (~2 GB PG-19 subset)
uv run python gpt.py --dataset shakespeare        # tiny corpus for quick experiments
uv run python gpt.py --dataset tinystories        # ~1.9 GB of simple short stories
```

Writes `./checkpoints/ckpt_<profile>_step_*.pt`. See `datasets/` for available corpora — each is a small subclass with its URL, local cache path, and (optionally) a `postprocess` method for cleaning the raw download.

At the end of training the final checkpoint is auto-uploaded to the HF Hub repo listed in `checkpoints.json` and tagged with the current git SHA (`ckpt_step_<step>_<sha>.pt`). A `dirty` suffix is appended if the working tree has uncommitted changes. Disable with `--no-upload`, or override the destination with `--upload-repo <user>/<repo>`. The script prints a JSON snippet you can paste into `checkpoints.json` to publish the upload.

## Pretrained checkpoints

The repo stays small; checkpoints live externally and are pinned by the git SHA of the training code. `checkpoints.json` is the manifest (step → URL + sha256), and `download_checkpoints.py` fetches them.

```bash
uv run python download_checkpoints.py                                # download all
uv run python download_checkpoints.py --profile default --step 4999  # download one
uv run python download_checkpoints.py --list                         # show what's available
```

Downloads verify sha256 and skip files that are already present.

## Training

`gpt.py` trains a character-level transformer on `input.txt` (auto-downloaded). Architecture, training schedule, and dropout are bundled into named `Hyperparameters` profiles at the top of `gpt.py`:

- `default` — n_embd 384, 6 layers, 6 heads, block_size 512, 5000 iters
- `tiny` — small enough to iterate on CPU-class setups
- `large` — 12 layers, 12 heads, block_size 1024

Switch profiles by editing `ACTIVE_PROFILE` in `gpt.py`. Each checkpoint embeds the architecture and vocab so it's self-contained for inference.

### Tokenizer

The training script supports two tokenizers:

```bash
uv run python gpt.py --tokenizer char                          # default; one token per character
uv run python gpt.py --tokenizer bpe --tokenizer-vocab-size 1024  # byte-level BPE (minbpe-style)
```

`char` keeps the original behavior: vocab = unique characters in the corpus (~85–250). `bpe` trains byte-pair merges on a 5 MB prefix of the corpus and gives you ~3–4× sequence compression on English, so `block_size=512` covers ~1.5–2k chars of real context. The encoded corpus is cached to disk (`{dataset_path}.{tokenizer}_v{vocab_size}.cache.pt`) keyed on the tokenizer state, since naive-Python BPE encoding of a 2 GB corpus takes minutes.

Checkpoints store the tokenizer state, so inference reconstructs the exact same tokenizer:

```bash
uv run python gpt.py --infer checkpoints/ckpt_default_step_04999.pt
#   tokenizer: bpe (vocab=1024)
```

Legacy checkpoints (only `chars` saved) load through a compatibility path as if they were `char` checkpoints — nothing breaks.

## Monitoring with TensorBoard

Each training run writes scalar loss curves and a 200-character text sample at every eval step to `./runs/<timestamp>_<profile>_<dataset>/`. Disable with `--no-tensorboard`, or keep the curves but skip generation with `--no-sample` (worth it on the `large` profile).

```bash
uv run tensorboard --logdir=runs
```

Opens at http://localhost:6006. To make it reachable from other machines on your Tailnet (or any LAN), bind to all interfaces:

```bash
uv run tensorboard --logdir=runs --bind_all
```

Then hit `http://<this-machine>:6006` from your phone/laptop over Tailscale.

## Browser-side inference (ONNX + GitHub Pages)

A trained checkpoint can be exported to ONNX and run entirely in the visitor's browser via ONNX Runtime Web — no server, no API key.

```bash
uv run python export_onnx.py checkpoints/ckpt_default_step_03375.pt
```

Writes `web/model.onnx` (~44 MB at the `default` profile) and `web/vocab.json`. The static frontend in `web/` loads both, drives the autoregressive sampling loop in JS, and renders the generated text live.

To preview locally:

```bash
python -m http.server -d web 8000
# open http://localhost:8000
```

To deploy: the `.github/workflows/pages.yml` workflow publishes `web/` to GitHub Pages on every push to master that touches `web/**`. Enable Pages in repo settings → Pages → Source: GitHub Actions, then merge a change. The model file is committed alongside the JS so the deploy is a single artifact.

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
- `tokenizers/` — char and byte-level BPE tokenizer classes (local package; not HuggingFace's)
- `bigram.py` — the tiny bigram baseline from earlier in the lecture
- `viz_embeddings.py` — Streamlit embedding viewer
- `checkpoint_io.py` — shared helpers for checkpoint files (sha256, git SHA tag, HF Hub upload, manifest)
- `checkpoints.json` + `download_checkpoints.py` — manifest and downloader for published checkpoints
- `checkpoints/` — training checkpoints (gitignored; populated by training or by the downloader)
- `runs/` — TensorBoard event files, one subdir per training run (gitignored)
- `export_onnx.py` — convert a `.pt` checkpoint to ONNX for browser-side inference
- `web/` — static HTML/JS demo (ONNX Runtime Web). Deployed to GitHub Pages on push

## Publishing new checkpoints (maintainers)

Checkpoints are hosted on a Hugging Face Hub model repo. `huggingface_hub` is already a regular dep, so its `hf` CLI is available via `uv run`. To publish a fresh batch:

```bash
# one-time: authenticate
uv run hf auth login

# create the model repo (one-time)
uv run hf repo create ng-video-lecture-checkpoints --type model

# copy & rename with profile + current git SHA so files are pinned to the training code
GIT_SHA=$(git rev-parse --short HEAD)
PROFILE=default
mkdir -p upload
for step in 0 100 500 1000 2000 4999; do
  s=$(printf "%05d" $step)
  cp "checkpoints/ckpt_${PROFILE}_step_${s}.pt" \
     "upload/ckpt_${PROFILE}_step_${s}_${GIT_SHA}.pt"
done

# upload
uv run hf upload scheuclu/ng-video-lecture-checkpoints ./upload .

# then edit checkpoints.json: bump filenames + sha256s for each step
```

## Notes from Karpathy's original README

Sadly the video lecture did not go too deep into model initialization, which is quite important for good performance. The code in this repo will train fine, but its convergence is slower because it starts off in a not-great spot in the weight space. See [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)'s `_init_weights` for the canonical version.

## License

MIT
