// The model in a worker, so that the page stays responsive while it runs.
// Messages in: { type: "load" }, { type: "run", ids }, { type: "generate",
// ids, steer, n }. Messages out: progress, ready, record, token, done, error.
// The model runs on the GPU (WebGPU) if the browser has an adapter, else on
// the CPU (WebAssembly).
import createModule from "./engine.mjs";
import { createEngine, MODEL_URL } from "./model.mjs";

const CACHE = "neurons-smollm2-135m";
let engine = null;

// The response from the Cache API, else from the network, with a copy into
// the cache. Without a Cache API (some private windows) each visit fetches.
async function cached(url) {
  let cache = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) return hit;
  } catch {
    cache = null;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  if (cache) cache.put(url, res.clone()).catch(() => {});
  return res;
}

async function* body(res) {
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    yield value;
  }
}

function argmax(xs) {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i;
  return best;
}

// A WebGPU device if the browser has an adapter, else null (the CPU path).
async function gpuDevice() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    return adapter ? await adapter.requestDevice() : null;
  } catch {
    return null;
  }
}

async function load() {
  const config = await (await cached(MODEL_URL + "config.json")).json();
  const tokenizer = await (await cached(MODEL_URL + "tokenizer.json")).json();
  const res = await cached(MODEL_URL + "model.safetensors");
  const total = Number(res.headers.get("content-length")) || 269060552;
  engine = await createEngine(createModule, await gpuDevice());
  await engine.load(config, body(res), (bytes) => postMessage({ type: "progress", bytes, total }));
  postMessage({ type: "ready", config, tokenizer, path: engine.path,
    maxTokens: engine.maxTokens(), maxContext: engine.maxContext() });
}

// One message at a time: a GPU forward waits for the GPU, and the next
// message must not start in that wait.
let queue = Promise.resolve();
onmessage = ({ data }) => { queue = queue.then(() => handle(data)); };

async function handle(data) {
  try {
    if (data.type === "load") await load();
    else if (data.type === "run") {
      // The neuron values of every position, [position][layer][neuron].
      engine.setSteer([]);
      engine.setRecord(true);
      await engine.forward(data.ids, 0);
      engine.setRecord(false);
      const per = engine.config.num_hidden_layers * engine.config.intermediate_size;
      const record = new Float32Array(data.ids.length * per);
      for (let p = 0; p < data.ids.length; p++) record.set(engine.neuronsAt(p), p * per);
      postMessage({ type: "record", record }, [record.buffer]);
    } else if (data.type === "generate") {
      // Greedy tokens after ids, with the steered neurons set at every position.
      engine.setSteer(data.steer);
      let logits = await engine.forward(data.ids, 0);
      let pos = data.ids.length;
      for (let k = 0; k < data.n && pos < engine.maxContext(); k++) {
        const next = argmax(logits);
        postMessage({ type: "token", token: next });
        if (k + 1 < data.n) logits = await engine.forward([next], pos++);
      }
      postMessage({ type: "done" });
    }
  } catch (e) {
    postMessage({ type: "error", message: String(e.message || e) });
  }
}
