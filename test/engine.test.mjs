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

const engine = await createEngine(createModule);
await engine.load(expect.config, chunks(997));
const neuronsOf = (n, start) => {
  const out = [];
  for (let p = start; p < start + n; p++) out.push(...engine.neuronsAt(p));
  return out;
};

test("the C and the wasm come from engine/llama.goose", () => {
  for (const line of readFileSync(new URL("gen/sha256sums", root), "utf8").trim().split("\n")) {
    const [sum, file] = line.split(/\s+/);
    assert.equal(createHash("sha256").update(readFileSync(new URL(file, root))).digest("hex"), sum, file);
  }
});

test("logits and neuron values of a prompt", () => {
  engine.setSteer([]);
  engine.setRecord(true);
  close(engine.forward(expect.prompt, 0), expect.logits, "logits");
  close(neuronsOf(expect.prompt.length, 0), expect.neurons, "neurons");
});

test("more tokens after the key-value cache", () => {
  engine.forward(expect.prompt, 0);
  close(engine.forward(expect.more, expect.prompt.length), expect.more_logits, "logits");
  close(neuronsOf(expect.more.length, expect.prompt.length), expect.more_neurons, "neurons");
});

test("one token at a time gives the same logits", () => {
  const all = [...expect.prompt, ...expect.more];
  let logits;
  for (let p = 0; p < all.length; p++) logits = engine.forward([all[p]], p);
  close(logits, expect.more_logits, "logits");
});

test("a steered neuron", () => {
  engine.setSteer([expect.steer]);
  close(engine.forward(expect.prompt, 0), expect.steer_logits, "logits");
  engine.setSteer([]);
});

test("calls that do not fit are refused", () => {
  assert.throws(() => engine.forward([expect.config.vocab_size], 0), /does not fit/);
  assert.throws(() => engine.forward([1], engine.maxContext()), /does not fit/);
  assert.throws(() => engine.setSteer([{ layer: 9, index: 0, value: 1 }]), /cannot steer/);
});
