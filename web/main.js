// Client-side inference for the nanoGPT char-level model.
// Loads the ONNX model + vocab metadata, then runs the autoregressive
// generation loop in JS. The model graph emits next-token probs already
// softmaxed; we just need to sample and append.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/+esm";

const $ = (id) => document.getElementById(id);
const status = $("status");
const output = $("output");
const stats = $("stats");
const goBtn = $("go");

let session = null;
let chars = null;
let stoi = null;
let blockSize = 0;

async function init() {
  try {
    status.textContent = "loading vocab…";
    const meta = await (await fetch("./vocab.json")).json();
    chars = meta.chars;
    stoi = Object.fromEntries(chars.map((c, i) => [c, i]));
    blockSize = meta.block_size;

    status.textContent = `loading model (this is the big one — ~${"40 MB"})…`;
    // Prefer WebGPU; fall back to WASM.
    session = await ort.InferenceSession.create("./model.onnx", {
      executionProviders: ["webgpu", "wasm"],
    });
    const backend = session.handler?._sessionOptions?.executionProviders?.[0] || "wasm";
    status.textContent = `ready · vocab=${chars.length} · block_size=${blockSize} · backend=${backend}`;
    goBtn.textContent = "generate";
    goBtn.disabled = false;
  } catch (err) {
    status.textContent = `init failed: ${err.message}`;
    console.error(err);
  }
}

// Sample with temperature + top-k from a probability array.
function sample(probs, temperature, topK) {
  // Convert probs back to logits so we can apply temperature, then re-normalize.
  // Pre-softmax temperature is the same as log-then-scale-then-softmax.
  const logits = new Float32Array(probs.length);
  for (let i = 0; i < probs.length; i++) {
    logits[i] = Math.log(Math.max(probs[i], 1e-12)) / temperature;
  }
  // Argsort desc, keep top-k.
  const indexed = Array.from(logits, (l, i) => [l, i]);
  indexed.sort((a, b) => b[0] - a[0]);
  const k = Math.min(topK, indexed.length);
  // Softmax over the k kept logits (subtract max for numerical stability).
  const maxL = indexed[0][0];
  let sumE = 0;
  const expScores = new Float32Array(k);
  for (let i = 0; i < k; i++) {
    expScores[i] = Math.exp(indexed[i][0] - maxL);
    sumE += expScores[i];
  }
  // Sample.
  const r = Math.random() * sumE;
  let cum = 0;
  for (let i = 0; i < k; i++) {
    cum += expScores[i];
    if (r < cum) return indexed[i][1];
  }
  return indexed[k - 1][1];
}

async function generate(nTokens, temperature, topK) {
  goBtn.disabled = true;
  output.textContent = "";
  stats.textContent = "";
  // Seed with the first vocab char as the starting context (token id 0).
  let tokens = [0];
  const t0 = performance.now();

  for (let step = 0; step < nTokens; step++) {
    const ctx = tokens.slice(-blockSize);
    const idx = new ort.Tensor(
      "int64",
      BigInt64Array.from(ctx, (v) => BigInt(v)),
      [1, ctx.length],
    );
    const out = await session.run({ idx });
    const probs = out.probs.data; // Float32Array, length vocab_size

    const next = sample(probs, temperature, topK);
    tokens.push(next);
    output.textContent += chars[next];

    // Yield to the UI thread every few tokens so the page stays responsive.
    if (step % 8 === 0) await new Promise((r) => setTimeout(r, 0));
  }

  const elapsed = (performance.now() - t0) / 1000;
  stats.textContent = `${nTokens} tokens in ${elapsed.toFixed(2)}s · ${(nTokens / elapsed).toFixed(1)} tok/s`;
  goBtn.disabled = false;
}

goBtn.addEventListener("click", () => {
  const n = parseInt($("ntokens").value, 10);
  const t = parseFloat($("temp").value);
  const k = parseInt($("topk").value, 10);
  generate(n, t, k).catch((err) => {
    status.textContent = `error: ${err.message}`;
    goBtn.disabled = false;
    console.error(err);
  });
});

init();
