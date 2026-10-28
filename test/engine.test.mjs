// The wasm engine against transformers, on the tiny model of make_fixture.py.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import createModule from "../web/engine.mjs";
import { createEngine } from "../web/model.mjs";

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

// The largest absolute difference between the GPU and the CPU path that the
// tests accept, in the logits and in the neuron values (README, Tests).
const TOLERANCE = 1e-4;

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

  if (path === "gpu") test("gpu: a WebGPU adapter", () => assert.ok(engine, "no WebGPU adapter"));

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

// The fixed prompts of the fixture, unsteered and steered: the largest
// absolute difference between the paths, in the logits and the neuron values.
if ("gpu" in engines) test("gpu: the same values as the cpu", { skip: !engines.gpu }, async () => {
  const worst = { logits: 0, neurons: 0 };
  const diff = (what, a, b) => {
    for (let i = 0; i < a.length; i++) worst[what] = Math.max(worst[what], Math.abs(a[i] - b[i]));
  };
  for (const steer of [[], [expect.steer]]) {
    const runs = [];
    for (const engine of [engines.cpu, engines.gpu]) {
      engine.setSteer(steer);
      engine.setRecord(true);
      const logits = [(await engine.forward(expect.prompt, 0)).slice()];
      logits.push((await engine.forward(expect.more, expect.prompt.length)).slice());
      const neurons = [];
      for (let p = 0; p < expect.prompt.length + expect.more.length; p++) neurons.push(engine.neuronsAt(p));
      runs.push({ logits, neurons });
      engine.setSteer([]);
    }
    runs[0].logits.forEach((l, i) => diff("logits", l, runs[1].logits[i]));
    runs[0].neurons.forEach((n, i) => diff("neurons", n, runs[1].neurons[i]));
  }
  assert.ok(worst.logits <= TOLERANCE && worst.neurons <= TOLERANCE, JSON.stringify(worst));
});
