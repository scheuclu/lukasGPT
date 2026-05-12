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

from gpt import load_model_from_checkpoint

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
    return sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_step_*.pt")))


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
def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    chars: list[str] = ckpt["chars"]
    sd = ckpt["model"]
    tok = sd["token_embedding_table.weight"].float()
    pos = sd["position_embedding_table.weight"].float()
    return chars, tok, pos


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, chars, hp = load_model_from_checkpoint(path, device=device)
    return model, chars, hp, device


@torch.no_grad()
def residuals_for_prompts(model, chars, device, prompts, layer_idx):
    """Encode each prompt, run the model, and return the residual at
    `layer_idx` at the last position of the prompt.

    Returns (matrix of shape (n_valid, n_embd), list of valid prompts,
    list of skipped (prompt, reason)).
    """
    stoi = {c: i for i, c in enumerate(chars)}
    block_size = model.hp.block_size
    vecs = []
    valid = []
    skipped = []
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


def prompt_scatter(coords: torch.Tensor, prompts: list[str]) -> go.Figure:
    hovers = [f"{p!r}" for p in prompts]
    short = [p if len(p) <= 20 else p[:18] + "…" for p in prompts]
    fig = go.Figure(data=go.Scatter3d(
        x=coords[:, 0].tolist(),
        y=coords[:, 1].tolist(),
        z=coords[:, 2].tolist(),
        mode="markers+text",
        text=short,
        textposition="top center",
        hovertext=hovers,
        hoverinfo="text",
        marker=dict(size=6, color="#1f77b4", line=dict(width=1, color="white")),
    ))
    fig.update_layout(template="plotly_white",
                      margin=dict(l=0, r=0, t=10, b=0),
                      height=700)
    fig.update_scenes(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
    return fig


def pca(x: torch.Tensor, k: int) -> torch.Tensor:
    x = x - x.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return x @ vh[:k].T


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
        format_func=lambda p: os.path.basename(p).replace("ckpt_step_", "step ").replace(".pt", ""),
    )

    st.header("Display")
    show_labels = st.checkbox("show character labels", value=False,
                              help="Useful in 2D; can be busy in 3D with 243 tokens.")
    selected_cats = st.multiselect(
        "categories",
        options=list(CATEGORY_COLORS.keys()),
        default=list(CATEGORY_COLORS.keys()),
    )

chars, tok_w, pos_w = load_checkpoint(ckpt_path)

c1, c2, c3 = st.columns(3)
c1.metric("vocab size", len(chars))
c2.metric("token emb dim", tok_w.shape[1])
c3.metric("block size", pos_w.shape[0])

tab3d, tab2d, tab_sim, tab_pos, tab_resid = st.tabs(
    ["3D tokens", "2D tokens", "Similarity heatmap", "Positions", "Residual stream"]
)

with tab3d:
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
    coords = pca(tok_w, 2)
    st.plotly_chart(
        token_scatter(coords, chars, dim=2,
                      selected_cats=selected_cats, show_labels=True),
        use_container_width=True,
    )
    st.caption("Same data as the 3D view, projected to the top-2 PCs.")

with tab_sim:
    st.plotly_chart(similarity_heatmap(tok_w, chars), use_container_width=True)
    st.caption("Cosine similarity between every pair of token embedding rows. "
               "Red = similar direction, blue = opposite. Hover for the pair "
               "and exact value.")

with tab_pos:
    pos_dim = st.radio("dimensions", options=[3, 2], horizontal=True, index=0)
    coords = pca(pos_w, pos_dim)
    st.plotly_chart(position_scatter(coords, dim=pos_dim),
                    use_container_width=True)
    st.caption("Learned position embeddings, colored by position index and "
               "connected in order. A smooth path means the model learned a "
               "continuous notion of position.")

with tab_resid:
    st.markdown(
        "Each prompt is encoded character-by-character and run through the "
        "model. We grab the residual stream at the chosen layer at the "
        "*last position* of the prompt — that's the vector the model would "
        "use to predict the next character, after attending over the whole "
        "prompt. Then we PCA-3D across prompts."
    )
    model, m_chars, m_hp, m_device = load_model(ckpt_path)

    default_prompts = (
        "the king\n"
        "the queen\n"
        "the brother\n"
        "the sister\n"
        "the boy\n"
        "the girl\n"
        "the man\n"
        "the woman\n"
        "the cat\n"
        "the dog"
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
            coords = pca(mat, 3)
            st.plotly_chart(prompt_scatter(coords, valid),
                            use_container_width=True)
            st.caption(
                f"Residual at layer {layer_idx}, last position. PCA-3D across "
                f"{mat.shape[0]} prompts. Pairs that should be semantically "
                "related (e.g. king/queen, brother/sister) ideally land near "
                "each other — but remember this is a 10M-param char model "
                "trained on ~1M chars, so the structure is weak and noisy."
            )
            st.markdown("##### Pairwise cosine similarity")
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
                "Cosine similarity between the full residual vectors. "
                "Red = aligned, blue = opposite, white = orthogonal. "
                "Transformer residuals are anisotropic (they cluster in a "
                "narrow cone), so the 'centered' mode is usually the most "
                "informative — it shows differences relative to the average "
                "prompt direction at this layer."
            )
        elif mat is not None:
            st.info("Need at least 2 valid prompts for PCA.")
