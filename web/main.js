// Client-side inference for the nanoGPT char-level model.
// Loads the ONNX model + vocab metadata, then runs the autoregressive
// generation loop in JS. The model graph emits next-token probs already
// softmaxed; we just need to sample and append.
//
// `ort` is the global from onnxruntime-web's script tag in index.html.
// We pin the WASM path to the same CDN — ORT's auto-discovery doesn't
// work reliably when the JS is loaded from a different origin.

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/";

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
    if (meta.checkpoint) {
      $("ckpt-info").textContent = `checkpoint: ${meta.checkpoint}`;
    }

    status.textContent = `loading model (this is the big one — ~${"40 MB"})…`;
    // Prefer WebGPU; fall back to WASM.
    session = await ort.InferenceSession.create("./model.onnx", {
      executionProviders: ["webgpu", "wasm"],
    });
    status.textContent = `ready · vocab=${chars.length} · block_size=${blockSize}`;
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

function encodePrompt(text) {
  const tokens = [];
  const missing = new Set();
  for (const ch of text) {
    if (ch in stoi) tokens.push(stoi[ch]);
    else missing.add(ch);
  }
  return { tokens, missing };
}

async function generate(nTokens, temperature, topK) {
  goBtn.disabled = true;
  output.innerHTML = "";
  stats.textContent = "";

  const promptText = $("prompt").value;
  const { tokens: promptTokens, missing } = encodePrompt(promptText);
  // Empty prompt: seed with the first vocab char (matches the CLI's
  // torch.zeros((1,1)) starting context).
  let tokens = promptTokens.length > 0 ? [...promptTokens] : [0];

  // Render the prompt prefix in a distinct color so it's clear what
  // the model generated vs what we primed it with.
  if (promptTokens.length > 0) {
    const primer = document.createElement("span");
    primer.className = "primer";
    primer.textContent = promptTokens.map((t) => chars[t]).join("");
    output.appendChild(primer);
  }
  if (missing.size > 0) {
    const oov = [...missing].map((c) => JSON.stringify(c)).join(", ");
    stats.textContent = `note: skipped ${missing.size} char(s) not in vocab: ${oov}`;
  }

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
    output.appendChild(document.createTextNode(chars[next]));

    // Yield to the UI thread every few tokens so the page stays responsive.
    if (step % 8 === 0) await new Promise((r) => setTimeout(r, 0));
  }

  const elapsed = (performance.now() - t0) / 1000;
  const perf = `${nTokens} tokens in ${elapsed.toFixed(2)}s · ${(nTokens / elapsed).toFixed(1)} tok/s`;
  stats.textContent = stats.textContent ? `${stats.textContent} · ${perf}` : perf;
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
