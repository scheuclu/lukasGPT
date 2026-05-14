"""Export a trained checkpoint to ONNX for browser-side inference.

The forward pass is wrapped to return *only* the last-position
softmax probabilities — that's all the autoregressive JS sampler
needs, and folding the softmax into the graph keeps the JS dead
simple.

Usage:
    uv run python export_onnx.py checkpoints/ckpt_default_step_03375.pt
"""

import argparse
import json
import os

import torch
import torch.nn as nn

from gpt import load_model_from_checkpoint


class InferenceWrapper(nn.Module):
    """Forward-only wrapper returning probabilities for the next token."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx)
        return torch.softmax(logits[:, -1, :], dim=-1)


def main() -> None:
    description = (__doc__ or "").splitlines()[0] if __doc__ else None
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("ckpt", help="Path to a .pt checkpoint.")
    ap.add_argument("--out", default="web/model.onnx", help="Output .onnx path.")
    ap.add_argument(
        "--vocab-out", default="web/vocab.json", help="Output vocab json path."
    )
    args = ap.parse_args()

    print(f"loading {args.ckpt}")
    model, chars, hp = load_model_from_checkpoint(args.ckpt, device="cpu")
    wrapped = InferenceWrapper(model)
    wrapped.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dummy = torch.zeros((1, min(8, hp.block_size)), dtype=torch.long)

    print(f"exporting → {args.out}")
    torch.onnx.export(
        wrapped,
        (dummy,),
        args.out,
        input_names=["idx"],
        output_names=["probs"],
        dynamic_axes={"idx": {1: "T"}, "probs": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"  wrote {args.out} ({size_mb:.1f} MB)")

    meta: dict[str, object] = {
        "chars": chars,
        "block_size": hp.block_size,
        "vocab_size": len(chars),
        "n_layer": hp.n_layer,
        "n_embd": hp.n_embd,
        "n_head": hp.n_head,
        "checkpoint": os.path.basename(args.ckpt),
    }
    with open(args.vocab_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {args.vocab_out}")


if __name__ == "__main__":
    main()
