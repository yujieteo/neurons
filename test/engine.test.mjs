// The wasm engine against transformers, on the tiny model of make_fixture.py.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import process from "node:process";
import createModule from "../web/engine.mjs";
import { createEngine } from "../web/model.mjs";
import { createTokenizer } from "../web/tokenizer.mjs";

const root = new URL("../", import.meta.url);
const expect = JSON.parse(readFileSync(new URL("test/fixture/tiny.json", root)));

// The file in small chunks, so that tensors and the header cross chunks.
async function* chunks(size) {
  const bytes = readFileSync(new URL("test/fixture/tiny.safetensors", root));
  for (let i = 0; i < bytes.length; i += size) yield bytes.subarray(i, i + size);
}

function close(actual, wanted, what) {
  assert.equal(actual.length, wanted.length, `${what}: length`);
  let worst = 0;
  for (let i = 0; i < wanted.length; i++) worst = Math.max(worst, Math.abs(actual[i] - wanted[i]) / (1e-3 + Math.abs(wanted[i])));
  assert.ok(worst < 1e-3, `${what}: worst relative error ${worst}`);
}

// The tolerance between the GPU and the CPU path (README, Tests): for each
// compared tensor, the logits of a forward call and its neuron values,
// max |gpu - cpu| <= TOLERANCE x max |cpu|.
const TOLERANCE = 1e-5;

// Runs each case { steer, calls: [[ids, start], ...] } on the cpu and the gpu
// engine. Returns the largest max |gpu - cpu| / max |cpu| of the logits and
// of the neuron values.
async function gpuError(cpu, gpu, cases) {
  const worst = { logits: 0, neurons: 0 };
  const compare = (what, a, b) => {
    let diff = 0, size = 0;
    for (let i = 0; i < a.length; i++) {
      diff = Math.max(diff, Math.abs(a[i] - b[i]));
      size = Math.max(size, Math.abs(a[i]));
    }
    worst[what] = Math.max(worst[what], diff && diff / size);
  };
  for (const { steer, calls } of cases) {
    for (const [ids, start] of calls) {
      const out = [];
      for (const engine of [cpu, gpu]) {
        engine.setSteer(steer);
        engine.setRecord(true);
        const logits = (await engine.forward(ids, start)).slice();
        const stride = engine.config.num_hidden_layers * engine.config.intermediate_size;
        const neurons = new Float32Array(ids.length * stride);
        for (let p = 0; p < ids.length; p++) neurons.set(engine.neuronsAt(start + p), p * stride);
        engine.setSteer([]);
        out.push({ logits, neurons });
      }
      compare("logits", out[0].logits, out[1].logits);
      compare("neurons", out[0].neurons, out[1].neurons);
    }
  }
  return worst;
}

// The CPU path always; the GPU path where the runtime has WebGPU (Deno, not
// Node). With WebGPU but no adapter, the GPU tests fail rather than skip.
const device = globalThis.navigator?.gpu
  ? await (await navigator.gpu.requestAdapter())?.requestDevice() ?? "no adapter"
  : null;
const engines = { cpu: await createEngine(createModule) };
if (device) engines.gpu = device === "no adapter" ? null : await createEngine(createModule, device);
for (const engine of Object.values(engines)) await engine?.load(expect.config, chunks(997));

test("the C and the wasm come from the engine sources", () => {
  for (const line of readFileSync(new URL("gen/sha256sums", root), "utf8").trim().split("\n")) {
    const [sum, file] = line.split(/\s+/);
    assert.equal(createHash("sha256").update(readFileSync(new URL(file, root))).digest("hex"), sum, file);
  }
});

for (const [path, engine] of Object.entries(engines)) {
  const neuronsOf = (n, start) => {
    const out = [];
    for (let p = start; p < start + n; p++) out.push(...engine.neuronsAt(p));
    return out;
  };

  if (path === "gpu") test("gpu: a WebGPU adapter that takes the weights", () => {
    assert.ok(engine, "no WebGPU adapter");
    assert.equal(engine.path, "gpu");
  });

  test(`${path}: logits and neuron values of a prompt`, { skip: !engine }, async () => {
    engine.setSteer([]);
    engine.setRecord(true);
    close(await engine.forward(expect.prompt, 0), expect.logits, "logits");
    close(neuronsOf(expect.prompt.length, 0), expect.neurons, "neurons");
  });

  test(`${path}: more tokens after the key-value cache`, { skip: !engine }, async () => {
    await engine.forward(expect.prompt, 0);
    close(await engine.forward(expect.more, expect.prompt.length), expect.more_logits, "logits");
    close(neuronsOf(expect.more.length, expect.prompt.length), expect.more_neurons, "neurons");
  });

  test(`${path}: one token at a time gives the same logits`, { skip: !engine }, async () => {
    const all = [...expect.prompt, ...expect.more];
    let logits;
    for (let p = 0; p < all.length; p++) logits = await engine.forward([all[p]], p);
    close(logits, expect.more_logits, "logits");
  });

  test(`${path}: a steered neuron`, { skip: !engine }, async () => {
    engine.setSteer([expect.steer]);
    close(await engine.forward(expect.prompt, 0), expect.steer_logits, "logits");
    engine.setSteer([]);
  });

  test(`${path}: calls that do not fit are refused`, { skip: !engine }, async () => {
    await assert.rejects(engine.forward([expect.config.vocab_size], 0), /does not fit/);
    await assert.rejects(engine.forward([1], engine.maxContext()), /does not fit/);
    assert.throws(() => engine.setSteer([{ layer: 9, index: 0, value: 1 }]), /cannot steer/);
  });
}

// The prompts of the fixture, unsteered and steered.
if ("gpu" in engines) test("gpu: the same values as the cpu", { skip: !engines.gpu }, async () => {
  const calls = [[expect.prompt, 0], [expect.more, expect.prompt.length]];
  const worst = await gpuError(engines.cpu, engines.gpu, [{ steer: [], calls }, { steer: [expect.steer], calls }]);
  assert.ok(worst.logits <= TOLERANCE && worst.neurons <= TOLERANCE, JSON.stringify(worst));
});

// SmolLM2-135M, where NEURONS_MODEL_DIR names a directory with its
// config.json, tokenizer.json and model.safetensors (the revision of
// MODEL_URL in web/model.mjs): 6 prompts, each unsteered and with 2 steering
// cases, and one more token after each.
const modelDir = process.env.NEURONS_MODEL_DIR;
if ("gpu" in engines) test("gpu: the same values as the cpu on SmolLM2-135M", { skip: !engines.gpu || !modelDir }, async () => {
  const file = (name) => readFileSync(`${modelDir}/${name}`);
  const config = JSON.parse(file("config.json"));
  const tokenizer = createTokenizer(JSON.parse(file("tokenizer.json")));
  const bytes = file("model.safetensors");
  async function* weights() {
    for (let i = 0; i < bytes.length; i += 1 << 24) yield bytes.subarray(i, i + (1 << 24));
  }
  const cpu = await createEngine(createModule);
  await cpu.load(config, weights());
  const gpu = await createEngine(createModule, await (await navigator.gpu.requestAdapter()).requestDevice());
  await gpu.load(config, weights());
  assert.equal(gpu.path, "gpu");
  const prompts = [
    "The capital of France is Paris, and the capital of Germany is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Photosynthesis is the process by which green plants use sunlight to",
    "In 1969, Neil Armstrong became the first person to walk on the",
    "Once upon a time, in a small village by the sea, there lived an old fisherman who",
    "The quick brown fox jumps over the lazy dog. ".repeat(12),
  ];
  const steers = [[], [{ layer: 10, index: 100, value: 8 }], [{ layer: 3, index: 7, value: -5 }, { layer: 25, index: 1000, value: 12 }]];
  const cases = [];
  for (const text of prompts) {
    const ids = tokenizer.encode(text).slice(0, 127);
    for (const steer of steers) cases.push({ steer, calls: [[ids, 0], [ids.slice(0, 1), ids.length]] });
  }
  const worst = await gpuError(cpu, gpu, cases);
  assert.ok(worst.logits <= TOLERANCE && worst.neurons <= TOLERANCE, JSON.stringify(worst));
});
