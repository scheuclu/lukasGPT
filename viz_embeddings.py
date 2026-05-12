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

tab3d, tab2d, tab_sim, tab_pos = st.tabs(
    ["3D tokens", "2D tokens", "Similarity heatmap", "Positions"]
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
