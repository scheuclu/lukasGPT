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
const promptPreview = $("prompt-preview");

let session = null;
let chars = null;
let stoi = null;
let blockSize = 0;

const TOK_COLORS = 6; // matches .tok-0..tok-5 in index.html

// Append one token as a colored <span> to `container`. colorIdx cycles
// through the six token-color CSS classes so adjacent tokens get
// distinguishable backgrounds. `primer=true` adds an underline so
// user-typed tokens are visually distinct from model-generated ones.
function appendTokenSpan(container, tokenId, colorIdx, primer) {
  const span = document.createElement("span");
  span.className =
    `tok tok-${colorIdx % TOK_COLORS}` + (primer ? " primer-tok" : "");
  span.textContent = chars[tokenId];
  container.appendChild(span);
}

// Re-render the small "tokenized prompt" preview below the input on
// every keystroke. Skips silently before the vocab has loaded.
function renderPromptPreview() {
  promptPreview.innerHTML = "";
  if (!stoi) return;
  const { tokens } = encodePrompt($("prompt").value);
  tokens.forEach((t, i) => appendTokenSpan(promptPreview, t, i, false));
}

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
    // Render the prompt preview now (vocab is enough), before the slow
    // ONNX model fetch. User sees colored tokens immediately rather
    // than waiting through the ~10s model load.
    renderPromptPreview();

    status.textContent = `loading model (this is the big one — ~${"40 MB"})…`;
    // Prefer WebGPU; fall back to WASM.
    session = await ort.InferenceSession.create("./model.onnx", {
      executionProviders: ["webgpu", "wasm"],
    });
    // Surface which fast paths are actually available. crossOriginIsolated
    // implies SharedArrayBuffer works, which lets ORT use multi-threaded
    // WASM; without it, ORT silently falls back to a slower single thread.
    const webgpu = "gpu" in navigator;
    const sab = self.crossOriginIsolated === true;
    status.textContent =
      `ready · vocab=${chars.length} · block_size=${blockSize}` +
      ` · webgpu=${webgpu} · cross-origin-isolated=${sab}`;
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

// Run the ONNX model on a single sequence (1, T) of token IDs and return
// next-token probabilities (Float32Array of length vocab_size).
async function forwardProbs(seq) {
  const ctx = seq.slice(-blockSize);
  const idx = new ort.Tensor(
    "int64",
    BigInt64Array.from(ctx, (v) => BigInt(v)),
    [1, ctx.length],
  );
  const out = await session.run({ idx });
  return out.probs.data;
}

// Single-step sampling: at each iteration, run the model once, sample one
// token with temperature + top-k. `cycle` is a {i: int} so the color
// counter keeps incrementing across the primer/generated boundary.
async function generateSingleStep(tokens, nTokens, temperature, topK, cycle) {
  for (let step = 0; step < nTokens; step++) {
    const probs = await forwardProbs(tokens);
    const next = sample(probs, temperature, topK);
    tokens.push(next);
    appendTokenSpan(output, next, cycle.i, false);
    cycle.i++;
    if (step % 8 === 0) await new Promise((r) => setTimeout(r, 0));
  }
}

// Lookahead sampling: at each round, expand a depth-`depth` tree with
// branching factor `width`, then sample one full path proportional to its
// joint (sum of per-step log) probability, and commit all of its tokens.
// Mirrors GPTLanguageModel.generate_lookahead in gpt.py.
async function generateLookahead(tokens, nTokens, temperature, depth, width, cycle) {
  let generated = 0;
  while (generated < nTokens) {
    const commit = Math.min(depth, nTokens - generated);
    let leaves = [tokens.slice()]; // each leaf: full token sequence so far
    let logProbs = [0.0]; // joint log-prob per leaf

    for (let level = 0; level < commit; level++) {
      const newLeaves = [];
      const newLogProbs = [];
      for (let i = 0; i < leaves.length; i++) {
        const probs = await forwardProbs(leaves[i]);
        // Top-`width` indices by probability.
        const indexed = Array.from(probs, (p, j) => [p, j]);
        indexed.sort((a, b) => b[0] - a[0]);
        const k = Math.min(width, indexed.length);
        for (let j = 0; j < k; j++) {
          newLeaves.push([...leaves[i], indexed[j][1]]);
          newLogProbs.push(
            logProbs[i] + Math.log(Math.max(indexed[j][0], 1e-12)),
          );
        }
      }
      leaves = newLeaves;
      logProbs = newLogProbs;
      await new Promise((r) => setTimeout(r, 0));
    }

    // Sample one path proportional to joint probability, with temperature.
    let maxLP = -Infinity;
    for (const lp of logProbs) if (lp > maxLP) maxLP = lp;
    const expScores = logProbs.map((lp) => Math.exp((lp - maxLP) / temperature));
    let sumE = 0;
    for (const e of expScores) sumE += e;
    const r = Math.random() * sumE;
    let cum = 0;
    let chosen = 0;
    for (let i = 0; i < expScores.length; i++) {
      cum += expScores[i];
      if (r < cum) { chosen = i; break; }
    }

    const path = leaves[chosen].slice(-commit);
    for (const t of path) {
      tokens.push(t);
      appendTokenSpan(output, t, cycle.i, false);
      cycle.i++;
    }
    generated += commit;
  }
}

async function generate(nTokens, temperature, topK, depth, width) {
  goBtn.disabled = true;
  output.innerHTML = "";
  stats.textContent = "";

  const promptText = $("prompt").value;
  const { tokens: promptTokens, missing } = encodePrompt(promptText);
  // Empty prompt: seed with the first vocab char (matches the CLI's
  // torch.zeros((1,1)) starting context).
  const tokens = promptTokens.length > 0 ? [...promptTokens] : [0];

  // Render each primer token as a colored span (with underline), and
  // keep a running color counter so generated tokens continue the
  // cycle without restarting at zero.
  const cycle = { i: 0 };
  for (const t of promptTokens) {
    appendTokenSpan(output, t, cycle.i, true);
    cycle.i++;
  }
  if (missing.size > 0) {
    const oov = [...missing].map((c) => JSON.stringify(c)).join(", ");
    stats.textContent = `note: skipped ${missing.size} char(s) not in vocab: ${oov}`;
  }

  const t0 = performance.now();
  if (depth > 1) {
    await generateLookahead(tokens, nTokens, temperature, depth, width, cycle);
  } else {
    await generateSingleStep(tokens, nTokens, temperature, topK, cycle);
  }

  const elapsed = (performance.now() - t0) / 1000;
  const mode = depth > 1 ? `lookahead d=${depth} w=${width}` : "single-step";
  const perf = `${nTokens} tokens in ${elapsed.toFixed(2)}s · ${(nTokens / elapsed).toFixed(1)} tok/s · ${mode}`;
  stats.textContent = stats.textContent ? `${stats.textContent} · ${perf}` : perf;
  goBtn.disabled = false;
}

// Lookahead is opt-in. When the checkbox is off, depth/width inputs are
// disabled and the single-step path runs regardless of their values.
const lookaheadBox = $("use-lookahead");
lookaheadBox.addEventListener("change", () => {
  $("depth").disabled = !lookaheadBox.checked;
  $("width").disabled = !lookaheadBox.checked;
});

// Live update the tokenized-prompt preview as the user types.
$("prompt").addEventListener("input", renderPromptPreview);

goBtn.addEventListener("click", () => {
  const n = parseInt($("ntokens").value, 10);
  const t = parseFloat($("temp").value);
  const k = parseInt($("topk").value, 10);
  const d = lookaheadBox.checked ? parseInt($("depth").value, 10) : 1;
  const w = parseInt($("width").value, 10);
  generate(n, t, k, d, w).catch((err) => {
    status.textContent = `error: ${err.message}`;
    goBtn.disabled = false;
    console.error(err);
  });
});

init();
