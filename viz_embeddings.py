"""Streamlit dashboard for learned token & position embeddings.

Run:
  uv run streamlit run viz_embeddings.py
"""

import glob
import os
import string

import plotly.graph_objects as go
import streamlit as st
import torch

from gpt import GPTLanguageModel, Hyperparameters, load_model_from_checkpoint
import tokenizers as tok
from tokenizers import CharTokenizer
from tokenizers.base import Tokenizer

CKPT_DIR = "checkpoints"

CATEGORY_COLORS = {
    "lowercase": "#1f77b4",
    "uppercase": "#2ca02c",
    "digit": "#d62728",
    "punctuation": "#ff7f0e",
    "whitespace": "#888888",
    "other": "#9467bd",
}


def list_checkpoints(ckpt_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))


def _format_ckpt(path: str) -> str:
    """Human-readable label for a checkpoint file. Supports both legacy
    `ckpt_step_<step>.pt` and the current `ckpt_<profile>_step_<step>.pt`."""
    name = os.path.basename(path).removeprefix("ckpt_").removesuffix(".pt")
    if name.startswith("step_"):
        return "step " + name.removeprefix("step_")
    profile, _, rest = name.partition("_step_")
    return f"{profile} · step {rest}"


def char_category(c: str) -> str:
    if c in " \t\n\r":
        return "whitespace"
    if c in string.digits:
        return "digit"
    if c in string.ascii_lowercase:
        return "lowercase"
    if c in string.ascii_uppercase:
        return "uppercase"
    if c in string.punctuation:
        return "punctuation"
    return "other"


def char_display(c: str) -> str:
    return {" ": "␠", "\n": "␤", "\t": "␉", "\r": "␍"}.get(c, c)


@st.cache_data(show_spinner=False)
def load_checkpoint(path: str) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # The embedding viz is char-tokenizer-only — per-character category
    # coloring and similarity heatmaps don't translate to BPE subword tokens.
    if "chars" in ckpt:
        chars: list[str] = ckpt["chars"]
    elif ckpt.get("tokenizer_type") == "char":
        chars = ckpt["tokenizer_state"]["chars"]
    else:
        raise ValueError(
            f"{os.path.basename(path)} is a non-char tokenizer "
            f"({ckpt.get('tokenizer_type', 'unknown')}); the embedding viz "
            f"only supports char-level checkpoints."
        )
    sd = ckpt["model"]
    tok = sd["token_embedding_table.weight"].float()
    pos = sd["position_embedding_table.weight"].float()
    return chars, tok, pos


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str) -> tuple[GPTLanguageModel, list[str], Hyperparameters, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, hp = load_model_from_checkpoint(path, device=device)
    if not isinstance(tokenizer, CharTokenizer):
        raise ValueError(
            f"{os.path.basename(path)} uses {tokenizer.name} tokenizer; "
            f"the embedding viz only supports char-level checkpoints."
        )
    return model, tokenizer.chars, hp, device


@st.cache_data(show_spinner=False)
def load_tokenizer(path: str) -> Tokenizer:
    """Lightweight loader: pulls just the tokenizer state out of a
    checkpoint without instantiating the model. Works for both char and
    BPE checkpoints, plus legacy `chars`-only files."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "tokenizer_type" in ckpt and "tokenizer_state" in ckpt:
        t = tok.get(ckpt["tokenizer_type"])
        t.load_state_dict(ckpt["tokenizer_state"])
    elif "chars" in ckpt:
        t = tok.get("char")
        t.load_state_dict({"chars": ckpt["chars"]})
    else:
        raise ValueError(f"{os.path.basename(path)} has no tokenizer info")
    return t


@torch.no_grad()
def residuals_for_prompts(
    model: GPTLanguageModel,
    chars: list[str],
    device: str,
    prompts: list[str],
    layer_idx: int,
) -> tuple[torch.Tensor | None, list[str], list[tuple[str, str]]]:
    """Encode each prompt, run the model, and return the residual at
    `layer_idx` at the last position of the prompt.

    Returns (matrix of shape (n_valid, n_embd), list of valid prompts,
    list of skipped (prompt, reason)).
    """
    stoi = {c: i for i, c in enumerate(chars)}
    block_size = model.hp.block_size
    vecs: list[torch.Tensor] = []
    valid: list[str] = []
    skipped: list[tuple[str, str]] = []
    for p in prompts:
        ids = [stoi[c] for c in p if c in stoi]
        if not ids:
            skipped.append((p, "no chars in this checkpoint's vocab"))
            continue
        ids = ids[-block_size:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        all_res = model.residuals(idx)
        vec = all_res[layer_idx][0, -1, :]
        vecs.append(vec.detach().cpu().float())
        valid.append(p)
    if not vecs:
        return None, [], skipped
    return torch.stack(vecs), valid, skipped


def prompt_similarity_heatmap(matrix: torch.Tensor, prompts: list[str],
                              mode: str = "centered") -> go.Figure:
    """Cosine-similarity heatmap.

    mode="centered" subtracts the mean residual across prompts before
    computing similarity. This removes the dominant shared "rogue
    direction" that makes raw transformer residuals look uniformly
    positive (anisotropy), revealing the relative structure.

    mode="raw" computes cosine on the original vectors; mostly red since
    residuals live in a narrow cone.

    mode="autoscale" is raw cosine but with the color range stretched to
    the off-diagonal min/max.
    """
    if mode == "centered":
        m = matrix - matrix.mean(0, keepdim=True)
    else:
        m = matrix
    w = torch.nn.functional.normalize(m, dim=1)
    sim = (w @ w.T).cpu().numpy()

    if mode == "autoscale":
        n = sim.shape[0]
        off = sim[~torch.eye(n, dtype=torch.bool).numpy()]
        zmin, zmax = float(off.min()), float(off.max())
    else:
        zmin, zmax = -1.0, 1.0

    short = [p if len(p) <= 24 else p[:22] + "…" for p in prompts]
    hover = [[f"{prompts[i]!r}<br>↔ {prompts[j]!r}<br>cos={sim[i, j]:.3f}"
              for j in range(len(prompts))] for i in range(len(prompts))]
    fig = go.Figure(data=go.Heatmap(
        z=sim, x=short, y=short,
        zmin=zmin, zmax=zmax, zmid=0 if zmin < 0 < zmax else None,
        colorscale="RdBu", reversescale=True,
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=max(400, 40 * len(prompts) + 100),
    )
    return fig


def pca(x: torch.Tensor, k: int) -> torch.Tensor:
    x = x - x.mean(0, keepdim=True)
    # torch.linalg.svd's stubs leak Unknown through the namedtuple unpacking.
    _, _, vh = torch.linalg.svd(x, full_matrices=False)  # pyright: ignore[reportUnknownVariableType]
    return x @ vh[:k].T  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]


def token_scatter(coords: torch.Tensor, chars: list[str], dim: int,
                  selected_cats: list[str], show_labels: bool) -> go.Figure:
    cats = [char_category(c) for c in chars]
    labels = [char_display(c) for c in chars]
    hovers = [f"char={c!r}<br>idx={i}<br>category={cat}"
              for i, (c, cat) in enumerate(zip(chars, cats))]

    fig = go.Figure()
    for cat in CATEGORY_COLORS:
        if cat not in selected_cats:
            continue
        idxs = [i for i, c in enumerate(cats) if c == cat]
        if not idxs:
            continue
        mode = "markers+text" if show_labels else "markers"
        kw = dict(
            mode=mode,
            name=cat,
            text=[labels[i] for i in idxs] if show_labels else None,
            textposition="top center" if dim == 2 else "middle center",
            hovertext=[hovers[i] for i in idxs],
            hoverinfo="text",
            marker=dict(size=6 if dim == 3 else 10,
                        color=CATEGORY_COLORS[cat],
                        line=dict(width=1, color="white")),
        )
        if dim == 2:
            fig.add_trace(go.Scatter(x=coords[idxs, 0].tolist(),
                                     y=coords[idxs, 1].tolist(), **kw))
        else:
            fig.add_trace(go.Scatter3d(x=coords[idxs, 0].tolist(),
                                       y=coords[idxs, 1].tolist(),
                                       z=coords[idxs, 2].tolist(), **kw))

    fig.update_layout(template="plotly_white",
                      legend=dict(itemsizing="constant"),
                      margin=dict(l=0, r=0, t=10, b=0),
                      height=700)
    if dim == 2:
        fig.update_xaxes(title="PC1")
        fig.update_yaxes(title="PC2", scaleanchor="x", scaleratio=1)
    else:
        fig.update_scenes(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
    return fig


def similarity_heatmap(weight: torch.Tensor, chars: list[str]) -> go.Figure:
    w = torch.nn.functional.normalize(weight, dim=1)
    sim = (w @ w.T).cpu().numpy()
    labels = [char_display(c) for c in chars]
    hover = [[f"{chars[i]!r} · {chars[j]!r}<br>cos={sim[i, j]:.3f}"
              for j in range(len(chars))] for i in range(len(chars))]
    fig = go.Figure(data=go.Heatmap(
        z=sim, x=labels, y=labels,
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(template="plotly_white",
                      yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
                      margin=dict(l=0, r=0, t=10, b=0),
                      height=750)
    return fig


def position_scatter(coords: torch.Tensor, dim: int) -> go.Figure:
    n = coords.shape[0]
    idx = list(range(n))
    common = dict(
        mode="markers+lines",
        marker=dict(size=4, color=idx, colorscale="Viridis",
                    colorbar=dict(title="position"), showscale=True),
        line=dict(color="rgba(120,120,120,0.3)", width=1),
        hovertext=[f"pos={i}" for i in idx],
        hoverinfo="text",
    )
    if dim == 2:
        fig = go.Figure(data=go.Scatter(
            x=coords[:, 0].tolist(), y=coords[:, 1].tolist(), **common))
        fig.update_xaxes(title="PC1")
        fig.update_yaxes(title="PC2", scaleanchor="x", scaleratio=1)
    else:
        fig = go.Figure(data=go.Scatter3d(
            x=coords[:, 0].tolist(), y=coords[:, 1].tolist(),
            z=coords[:, 2].tolist(), **common))
        fig.update_scenes(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
    fig.update_layout(template="plotly_white",
                      margin=dict(l=0, r=0, t=10, b=0),
                      height=700)
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="nanoGPT embedding viz", layout="wide")
st.title("Learned embedding explorer")

ckpts = list_checkpoints(CKPT_DIR)
if not ckpts:
    st.error(f"No checkpoints found in ./{CKPT_DIR}")
    st.stop()

with st.sidebar:
    st.header("Checkpoint")
    ckpt_path = st.selectbox(
        "step",
        options=ckpts,
        index=len(ckpts) - 1,
        format_func=lambda p: _format_ckpt(p),
    )

    st.header("Display")
    show_labels = st.checkbox("show character labels", value=False,
                              help="Useful in 2D; can be busy in 3D with 243 tokens.")
    selected_cats = st.multiselect(
        "categories",
        options=list(CATEGORY_COLORS.keys()),
        default=list(CATEGORY_COLORS.keys()),
    )

tokenizer = load_tokenizer(ckpt_path)

# Embedding-based tabs need the model weight matrices and are
# char-tokenizer-only. For BPE checkpoints we'll show the tokens tab
# anyway and disable the rest.
chars: list[str] | None = None
tok_w: torch.Tensor | None = None
pos_w: torch.Tensor | None = None
try:
    chars, tok_w, pos_w = load_checkpoint(ckpt_path)
except ValueError as e:
    st.warning(str(e))

c1, c2, c3 = st.columns(3)
c1.metric("vocab size", tokenizer.vocab_size)
c2.metric("tokenizer", tokenizer.name)
c3.metric("block size", pos_w.shape[0] if pos_w is not None else "?")

tab_tokens, tab3d, tab2d, tab_sim, tab_pos, tab_resid = st.tabs(
    ["Tokens", "3D tokens", "2D tokens", "Similarity heatmap", "Positions", "Residual stream"]
)


def _embedding_only(label: str) -> None:
    st.info(
        f"The **{label}** tab visualizes the learned token embedding "
        f"matrix, which only has a meaningful per-character interpretation "
        f"for `char` tokenizers. This checkpoint uses `{tokenizer.name}`."
    )


with tab_tokens:
    st.markdown(
        f"Vocabulary learned by this checkpoint's `{tokenizer.name}` tokenizer. "
        f"For BPE, each row is a byte sequence merged out of more frequent "
        f"adjacent pairs during training; for char, each row is one character "
        f"that appeared in the training corpus."
    )
    rows: list[dict[str, int | str]] = []
    for token_id in range(tokenizer.vocab_size):
        s = tokenizer.decode([token_id])
        rows.append({
            "id": token_id,
            "token": repr(s),
            "chars": len(s),
            "bytes": len(s.encode("utf-8")),
        })
    st.dataframe(rows, use_container_width=True, height=600, hide_index=True)
    if tokenizer.vocab_size > 256:
        # BPE token-length histogram — char-level is uninteresting (all 1s).
        from collections import Counter
        length_counts: Counter[int] = Counter(int(r["chars"]) for r in rows)
        xs: list[int] = sorted(length_counts)
        ys: list[int] = [length_counts[x] for x in xs]
        fig = go.Figure(data=go.Bar(x=xs, y=ys))
        fig.update_layout(template="plotly_white",
                          title="Token length distribution (chars)",
                          xaxis_title="length", yaxis_title="count",
                          height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

with tab3d:
    if chars is None or tok_w is None:
        _embedding_only("3D tokens")
    else:
        coords = pca(tok_w, 3)
        st.plotly_chart(
            token_scatter(coords, chars, dim=3,
                          selected_cats=selected_cats, show_labels=show_labels),
            use_container_width=True,
        )
        st.caption("Drag to rotate, scroll to zoom. Each point is one character "
                   "of the vocabulary projected into the top-3 principal components "
                   "of the learned token embedding matrix.")

with tab2d:
    if chars is None or tok_w is None:
        _embedding_only("2D tokens")
    else:
        coords = pca(tok_w, 2)
        st.plotly_chart(
            token_scatter(coords, chars, dim=2,
                          selected_cats=selected_cats, show_labels=True),
            use_container_width=True,
        )
        st.caption("Same data as the 3D view, projected to the top-2 PCs.")

with tab_sim:
    if chars is None or tok_w is None:
        _embedding_only("Similarity heatmap")
    else:
        st.plotly_chart(similarity_heatmap(tok_w, chars), use_container_width=True)
        st.caption("Cosine similarity between every pair of token embedding rows. "
                   "Red = similar direction, blue = opposite. Hover for the pair "
                   "and exact value.")

with tab_pos:
    if pos_w is None:
        _embedding_only("Positions")
    else:
        pos_dim = st.radio("dimensions", options=[3, 2], horizontal=True, index=0)
        coords = pca(pos_w, pos_dim)
        st.plotly_chart(position_scatter(coords, dim=pos_dim),
                        use_container_width=True)
        st.caption("Learned position embeddings, colored by position index and "
                   "connected in order. A smooth path means the model learned a "
                   "continuous notion of position.")

with tab_resid:
    if not isinstance(tokenizer, CharTokenizer):
        _embedding_only("Residual stream")
        st.stop()
    st.markdown(
        "Each prompt is encoded character-by-character and run through the "
        "model. We grab the residual stream at the chosen layer at the "
        "*last position* of the prompt — that's the vector the model would "
        "use to predict the next character, after attending over the whole "
        "prompt. Then we plot pairwise cosine similarity across prompts."
    )
    model, m_chars, m_hp, m_device = load_model(ckpt_path)

    default_prompts = (
        "mother\n"
        "brother\n"
        "sister\n"
        "father\n"
        "my fathers daughter\n"
        "sibling"
    )
    prompts_text = st.text_area(
        "prompts (one per line)", value=default_prompts, height=240,
        help="Short prompts work best. Each prompt must contain at least one "
             "character that exists in this checkpoint's vocab.",
    )
    prompts = [p for p in prompts_text.splitlines() if p.strip()]

    n_layers = m_hp.n_layer
    layer_labels = (
        ["0 · tok+pos embedding"]
        + [f"{i + 1} · after block {i}" for i in range(n_layers)]
        + [f"{n_layers + 1} · after final layer norm"]
    )
    layer_idx = st.select_slider(
        "layer",
        options=list(range(n_layers + 2)),
        value=n_layers + 1,
        format_func=lambda i: layer_labels[i],
    )

    if not prompts:
        st.info("Enter at least one prompt.")
    else:
        mat, valid, skipped = residuals_for_prompts(
            model, m_chars, m_device, prompts, layer_idx
        )
        if skipped:
            for p, reason in skipped:
                st.warning(f"skipped {p!r}: {reason}")
        if mat is not None and mat.shape[0] >= 2:
            sim_mode = st.radio(
                "color scale",
                options=["centered", "raw", "autoscale"],
                index=0, horizontal=True,
                help=(
                    "centered: subtract mean residual first (reveals real "
                    "structure by removing transformer anisotropy). "
                    "raw: cosine on original vectors, scale [-1, 1] — usually "
                    "uniformly red because residuals live in a narrow cone. "
                    "autoscale: raw cosine but color range stretched to the "
                    "off-diagonal min/max."
                ),
            )
            st.plotly_chart(
                prompt_similarity_heatmap(mat, valid, mode=sim_mode),
                use_container_width=True,
            )
            st.caption(
                f"Residual at layer {layer_idx}, last position. Cosine "
                f"similarity between the full residual vectors across "
                f"{mat.shape[0]} prompts. Red = aligned, blue = opposite, "
                "white = orthogonal. Transformer residuals are anisotropic "
                "(they cluster in a narrow cone), so the 'centered' mode is "
                "usually the most informative — it shows differences relative "
                "to the average prompt direction at this layer."
            )
        elif mat is not None:
            st.info("Need at least 2 valid prompts for PCA.")
