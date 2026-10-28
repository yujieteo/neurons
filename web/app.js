// The page: the snapshot of the sweep, the model in a worker, the layers,
// steering, and Claude through the visitor's own OpenRouter key.
import { createLayers } from "./art.mjs";
import { createTokenizer } from "./tokenizer.mjs";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids);
  return e;
};
const fmt = (x, d = 2) => (Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(d)).replace("-", "−");
const show = (t) => t.replace(/\n/g, "↵").replace(/\t/g, "⇥");

// ---- the snapshot ----
const [snap, bin] = await Promise.all([
  fetch("snapshot/explored.json").then((r) => r.json()),
  fetch("snapshot/neurons.bin").then((r) => r.arrayBuffer()),
]);
const L = snap.meta.layers, F = snap.meta.neurons_per_layer, N = L * F;
const stats = new Float32Array(bin, 0, N * 3);            // [mean, std, max] per neuron
const topIds = new Uint16Array(bin, N * 12, N * 3);       // three peak tokens per neuron
const explored = new Map(snap.neurons.map((n) => [`${n.layer}.${n.index}`, n]));
const mean = (l, f) => stats[(l * F + f) * 3];
const std = (l, f) => stats[(l * F + f) * 3 + 1];
const largest = (l, f) => stats[(l * F + f) * 3 + 2];
const zOf = (l, f, v) => (v - mean(l, f)) / (std(l, f) || 1);
const peakTokens = (l, f) => [...new Set(Array.from(topIds.subarray((l * F + f) * 3, (l * F + f) * 3 + 3)))]
  .map((id) => snap.vocab[id]);

$("nTotal").textContent = N.toLocaleString("en");
$("nTokens").textContent = `${(snap.meta.tokens / 1e6).toLocaleString("en", { maximumFractionDigits: 1 })} million`;
$("nExplored").textContent = snap.neurons.length.toLocaleString("en");

// ---- state ----
let selected = null;           // { layer, index }
let live = null;               // { ids, pieces, record }
let livePos = 0;
let tokenizer = null, maxTokens = 128;
let lastSteer = null;          // { prompt, value, base, steered } of the last Generate
const per = L * F;
const liveValue = (p, l, f) => live.record[p * per + l * F + f];

const layers = createLayers($("layers"), L, (layer, index) => select(layer, index));

// ---- the model ----
const worker = new Worker(new URL("worker.mjs", import.meta.url), { type: "module" });
let onToken = null, onDone = null, onFail = null;
worker.onmessage = ({ data }) => {
  if (data.type === "progress") {
    $("modelStatus").textContent = `Loading the model: ${(data.bytes / 1e6).toFixed(0)} of ${(data.total / 1e6).toFixed(0)} MB`;
  } else if (data.type === "ready") {
    tokenizer = createTokenizer(data.tokenizer);
    maxTokens = data.maxTokens;
    $("modelStatus").textContent = "The model is ready. It runs in this tab.";
    $("load").hidden = true;
    $("run").disabled = false;
    $("generate").disabled = !selected;
  } else if (data.type === "record") {
    live.record = data.record;
    $("run").disabled = false;
    pickToken(live.ids.length - 1);
  } else if (data.type === "token") {
    onToken?.(data.token);
  } else if (data.type === "done") {
    onDone?.();
  } else if (data.type === "error") {
    $("modelStatus").textContent = `Error: ${data.message}`;
    onFail?.();
    $("load").disabled = !!tokenizer;
    $("run").disabled = !tokenizer;
    $("generate").disabled = !tokenizer || !selected;
  }
};

$("load").onclick = () => {
  $("load").disabled = true;
  $("modelStatus").textContent = "Loading the model…";
  worker.postMessage({ type: "load" });
};

$("run").onclick = () => {
  const ids = tokenizer.encode($("text").value).slice(0, maxTokens);
  if (!ids.length) return;
  live = { ids, pieces: ids.map((id) => tokenizer.decode([id])), record: null };
  $("run").disabled = true;
  worker.postMessage({ type: "run", ids });
};

// ---- your text ----
function pickToken(p) {
  livePos = p;
  renderTokens();
  // The strongest neurons at the token: by z-score, over all layers.
  const zs = new Float32Array(per);
  for (let l = 0; l < L; l++) for (let f = 0; f < F; f++) zs[l * F + f] = zOf(l, f, liveValue(p, l, f));
  const order = Array.from(zs.keys()).sort((a, b) => zs[b] - zs[a]);
  const points = [], path = [];
  for (let l = 0; l < L; l++) {
    const best = Array.from({ length: F }, (_, f) => f).sort((a, b) => zs[l * F + b] - zs[l * F + a]).slice(0, 6);
    for (const f of best) points.push({ layer: l, index: f, z: zs[l * F + f], mark: explored.has(`${l}.${f}`) });
    path.push({ layer: l, index: best[0], z: zs[l * F + best[0]] });
  }
  layers.show(points, path, layers.bins(zs));
  $("layersCaption").textContent = `The neurons at “${show(live.pieces[p])}”. Each rule is a layer, from the first `
    + "at the left. Its six strongest neurons are points, as high as their value stands above their mean in the sweep, in the accent if Claude explored them. "
    + "The path runs through the strongest neuron of each layer. At the right, all the values against the normal density.";
  $("topTitle").hidden = false;
  $("topToken").textContent = show(live.pieces[p]);
  $("top").replaceChildren(...order.slice(0, 12).map((i) => neuronItem(Math.floor(i / F), i % F, `z ${fmt(zs[i], 1)}`)));
  if (selected) {
    renderNeuron();
    layers.select({ ...selected, z: zs[selected.layer * F + selected.index] });
  }
}

function renderTokens() {
  const box = $("tokens");
  if (!live?.record) { box.replaceChildren(); return; }
  const s = selected;
  const scale = s ? Math.max(largest(s.layer, s.index), 1e-6) : 1;
  box.replaceChildren(...live.pieces.map((piece, p) => {
    const b = el("button", { className: "tok" + (p === livePos ? " at" : ""), textContent: show(piece) });
    if (s) {
      const v = liveValue(p, s.layer, s.index);
      b.style.setProperty("--a", Math.max(0, Math.min(1, v / scale)).toFixed(3));
      b.title = `value ${fmt(v)}`;
    }
    b.onclick = () => pickToken(p);
    return b;
  }));
}

// ---- neurons ----
function neuronItem(l, f, extra) {
  const n = explored.get(`${l}.${f}`);
  const b = el("button", { className: "item" + (n ? " known" : "") },
    el("span", { className: "id", textContent: `${l}.${f}` }),
    el("span", { className: "what", textContent: n ? n.label : peakTokens(l, f).map((t) => `“${show(t)}”`).join(" ") }),
    extra ? el("span", { className: "z", textContent: extra }) : "");
  b.onclick = () => select(l, f);
  return el("li", {}, b);
}

function contextLine(tokens, values, peak, scale) {
  return el("li", {}, el("span", { className: "v", textContent: fmt(values[peak]) }),
    el("span", { className: "ctx" }, ...tokens.map((t, i) => {
      const s = el("span", { textContent: show(t), title: fmt(values[i]) });
      s.style.setProperty("--a", Math.max(0, Math.min(1, values[i] / scale)).toFixed(3));
      if (i === peak) s.className = "peak";
      return s;
    })));
}

function select(layer, index, scroll = true) {
  selected = { layer, index };
  location.hash = `${layer}.${index}`;
  renderNeuron();
  renderTokens();
  layers.select({ layer, index, z: live?.record ? zOf(layer, index, liveValue(livePos, layer, index)) : zOf(layer, index, largest(layer, index)) });
  $("generate").disabled = !tokenizer;
  const box = document.querySelector(".neuron").getBoundingClientRect();
  if (scroll && (box.top < 0 || box.top > innerHeight)) document.querySelector(".neuron").scrollIntoView({ block: "start" });
}

function renderNeuron() {
  const { layer, index } = selected, n = explored.get(`${layer}.${index}`);
  $("nId").textContent = `Layer ${layer} · neuron ${index}`;
  $("nLabel").textContent = n ? n.label : "Not explored";
  $("nDesc").textContent = n ? n.description
    : `Claude did not explore this neuron. Its strongest tokens in the sweep: ${peakTokens(layer, index).map((t) => `“${show(t)}”`).join(", ")}. Run your own text, steer it, or ask Claude about it.`;
  $("nStats").textContent = `Largest value ${fmt(largest(layer, index))}, mean ${fmt(mean(layer, index), 3)}, `
    + `standard deviation ${fmt(std(layer, index), 3)} over the sweep.`;
  if (live?.record) {
    $("nLive").replaceChildren(el("h3", { textContent: "On your text" }),
      el("ol", { className: "contexts" }, contextLine(live.pieces,
        live.pieces.map((_, p) => liveValue(p, layer, index)), livePos, Math.max(largest(layer, index), 1e-6))));
  } else $("nLive").replaceChildren();
  const scale = n ? n.max : 1;
  $("nExamples").replaceChildren(...(n ? n.examples.map((e) => contextLine(e.tokens, e.values, e.peak, scale))
    : [el("li", { className: "note hint", textContent: "The snapshot keeps contexts for the explored neurons only." })]));
  $("nSteer").replaceChildren(...(n ? [el("table", {},
    el("thead", {}, el("tr", {}, el("th", { textContent: "Value" }), el("th", { textContent: "Continuation" }))),
    el("tbody", {}, ...snap.meta.prompts.map((p, i) => el("tr", {}, el("td", { textContent: "none" }),
      el("td", {}, el("b", { textContent: p }), document.createTextNode(snap.meta.baseline[i])))),
    ...n.steer.map((s) => el("tr", {}, el("td", { textContent: `${s.strength} × ${fmt(n.max)}` }),
      el("td", {}, el("b", { textContent: s.prompt }), document.createTextNode(s.text))))))] : []));
  updateSteerOut();
}

// ---- steering ----
function updateSteerOut() {
  const k = Number($("steerValue").value);
  $("steerOut").textContent = selected ? `${k.toFixed(1)} × ${fmt(largest(selected.layer, selected.index))} = ${fmt(k * largest(selected.layer, selected.index))}` : "";
}
$("steerValue").oninput = updateSteerOut;

function generate(ids, steer, out) {
  return new Promise((resolve, reject) => {
    const got = [];
    onToken = (t) => { got.push(t); out.textContent = tokenizer.decode(got); };
    onDone = () => resolve(tokenizer.decode(got));
    onFail = reject;
    worker.postMessage({ type: "generate", ids, steer, n: 32 });
  });
}

$("generate").onclick = async () => {
  const { layer, index } = selected;
  const value = Number($("steerValue").value) * largest(layer, index);
  const prompt = $("steerPrompt").value;
  const ids = tokenizer.encode(prompt).slice(0, maxTokens);
  if (!ids.length) return;
  $("generate").disabled = $("run").disabled = true;
  $("genBase").textContent = $("genSteer").textContent = "";
  try {
    const base = await generate(ids, [], $("genBase"));
    const steered = await generate(ids, [{ layer, index, value }], $("genSteer"));
    lastSteer = { layer, index, prompt, value, base, steered };
    $("generate").disabled = $("run").disabled = false;
  } catch {}
};

// ---- search ----
function search() {
  const q = $("search").value.trim();
  const ql = q.toLowerCase();
  let items;
  const id = /^(\d+)\s*[.:,\s]\s*(\d+)$/.exec(q);
  if (id && +id[1] < L && +id[2] < F) items = [[+id[1], +id[2]]];
  else if (!q) items = snap.neurons.map((n) => [n.layer, n.index]);
  else {
    items = snap.neurons.filter((n) => (n.label + " " + n.description).toLowerCase().includes(ql)).map((n) => [n.layer, n.index]);
    // Then any neuron whose peak tokens hold the query as a whole token.
    for (let l = 0; l < L && items.length < 200; l++) for (let f = 0; f < F; f++) {
      if (!explored.has(`${l}.${f}`) && peakTokens(l, f).some((t) => t.trim().toLowerCase() === ql)) items.push([l, f]);
    }
  }
  $("results").replaceChildren(...items.slice(0, 200).map(([l, f]) => neuronItem(l, f)));
}
$("search").oninput = search;

// ---- Ask Claude ----
try { $("key").value = sessionStorage.getItem("openrouter-key") || ""; } catch { /* no storage */ }
$("key").onchange = () => { try { sessionStorage.setItem("openrouter-key", $("key").value.trim()); } catch { /* no storage */ } };

function evidence() {
  const { layer, index } = selected, n = explored.get(`${layer}.${index}`);
  const mark = (tokens, peak) => tokens.map((t, i) => (i === peak ? `[[${t}]]` : t)).join("");
  const lines = [
    `Model: SmolLM2-135M, ${L} layers, ${F} MLP neurons per layer. A neuron is one element of silu(gate_proj(x)) * up_proj(x), the input of down_proj.`,
    `Neuron: layer ${layer}, index ${index}. Over ${snap.meta.tokens} tokens of FineWeb-Edu and Cosmopedia: mean ${fmt(mean(layer, index), 4)}, standard deviation ${fmt(std(layer, index), 4)}, largest value ${fmt(largest(layer, index))}.`,
    `Most frequent tokens at its top contexts: ${peakTokens(layer, index).map((t) => JSON.stringify(t)).join(", ")}.`,
  ];
  if (n) {
    lines.push(`Earlier label: ${n.label}. ${n.description}`, "", "Top contexts (value, then the text with the peak token in [[ ]]):");
    for (const e of n.examples) lines.push(`${fmt(e.values[e.peak])}  ${JSON.stringify(mark(e.tokens, e.peak))}`);
    lines.push("", "Steering (the neuron set to the value at every position; greedy continuation):");
    snap.meta.prompts.forEach((p, i) => lines.push(`unsteered: ${JSON.stringify(p)} -> ${JSON.stringify(snap.meta.baseline[i])}`));
    for (const s of n.steer) lines.push(`value ${fmt(s.strength * n.max)}: ${JSON.stringify(s.prompt)} -> ${JSON.stringify(s.text)}`);
  }
  if (live?.record) {
    lines.push("", "The visitor's text, each token with this neuron's value:");
    lines.push(live.pieces.map((t, p) => `${JSON.stringify(t)}:${fmt(liveValue(p, layer, index))}`).join(" "));
  }
  if (lastSteer && lastSteer.layer === layer && lastSteer.index === index) {
    lines.push("", `The visitor steered it to ${fmt(lastSteer.value)}: ${JSON.stringify(lastSteer.prompt)}`,
      `unsteered -> ${JSON.stringify(lastSteer.base)}`, `steered -> ${JSON.stringify(lastSteer.steered)}`);
  }
  return lines.join("\n");
}

$("ask").onclick = async () => {
  const key = $("key").value.trim(), answer = $("answer");
  if (!selected) { answer.textContent = "Select a neuron first."; return; }
  if (!key) { answer.textContent = "Put your OpenRouter key in first."; return; }
  try { sessionStorage.setItem("openrouter-key", key); } catch { /* no storage */ }
  $("ask").disabled = true;
  answer.textContent = "";
  try {
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: { "Authorization": `Bearer ${key}`, "Content-Type": "application/json", "X-Title": "Neurons" },
      body: JSON.stringify({
        model: $("askModel").value, stream: true, max_tokens: 1200,
        messages: [
          { role: "system", content: "You interpret neurons of a small language model from evidence. Be concrete and brief. Say what the evidence does not show." },
          { role: "user", content: `${evidence()}\n\nQuestion: ${$("question").value}` },
        ],
      }),
    });
    if (!res.ok) throw new Error(`OpenRouter: HTTP ${res.status} ${(await res.text()).slice(0, 300)}`);
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "", text = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += value;
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        const msg = JSON.parse(line.slice(6));
        if (msg.error) throw new Error(msg.error.message);
        text += msg.choices?.[0]?.delta?.content ?? "";
        answer.replaceChildren(...text.split(/\n{2,}/).map((p) => el("p", { textContent: p })));
      }
    }
  } catch (e) {
    answer.textContent = String(e.message || e);
  } finally {
    $("ask").disabled = false;
  }
};

// ---- start ----
function idleLayers() {
  const points = snap.neurons.map((n) => ({ layer: n.layer, index: n.index, z: zOf(n.layer, n.index, n.max), mark: false }));
  const path = [];
  for (let l = 0; l < L; l++) {
    const best = points.filter((p) => p.layer === l).sort((a, b) => b.z - a.z)[0];
    if (best) path.push(best);
  }
  layers.show(points, path, null);
  $("layersCaption").textContent = "Each rule is a layer, from the first at the left. Each point is an explored neuron, "
    + "as high as its largest value stands above its mean in the sweep. Select one to open it, "
    + "or run your own text to see the neurons at a token.";
}

idleLayers();
search();
const hash = /^#(\d+)\.(\d+)$/.exec(location.hash);
if (hash && +hash[1] < L && +hash[2] < F) select(+hash[1], +hash[2], false);
else select(snap.neurons[0].layer, snap.neurons[0].index, false);
