// The Jacobian lens and the neuron swap (README, Lens and swap), on an
// engine of model.mjs. Both change one neuron at one token: they run the text
// up to that token, then run that token again with the neuron set.

// Runs ids[0..pos] with recording on. Returns the logits at pos and the
// neuron values at pos, [layer][neuron]. The key-value cache then holds the
// positions before pos, unchanged by any later call at pos.
async function runTo(engine, ids, pos) {
  engine.setSteer([]);
  engine.setRecord(true);
  const logits = (await engine.forward(ids.slice(0, pos + 1), 0)).slice();
  return { logits, neurons: engine.neuronsAt(pos) };
}

// Runs token ids[pos] at pos again, with neuron (layer, index) set to value
// at that token only.
async function runSet(engine, ids, pos, layer, index, value) {
  engine.setSteer([{ layer, index, value }]);
  engine.setRecord(true);
  const logits = (await engine.forward([ids[pos]], pos)).slice();
  const neurons = engine.neuronsAt(pos);
  engine.setSteer([]);
  engine.setRecord(false);
  return { logits, neurons };
}

// The Jacobian lens of neuron (layer, index) at token pos of ids: the
// derivatives of the logits at pos (dLogits, [vocab]) and of the neuron
// values at pos (dNeurons, [layer][neuron]) with respect to the neuron's
// value at pos, by central differences with step h. Also the logits and the
// neuron values at pos without the change.
export async function jacobianLens(engine, ids, pos, layer, index, h) {
  const base = await runTo(engine, ids, pos);
  const value = base.neurons[layer * engine.config.intermediate_size + index];
  // The steps that the engine uses, as f32.
  const up = Math.fround(value + h), down = Math.fround(value - h);
  const plus = await runSet(engine, ids, pos, layer, index, up);
  const minus = await runSet(engine, ids, pos, layer, index, down);
  const diff = (a, b) => a.map((x, i) => (x - b[i]) / (up - down));
  return { value, logits: base.logits, neurons: base.neurons,
    dLogits: diff(plus.logits, minus.logits), dNeurons: diff(plus.neurons, minus.neurons) };
}

// The neuron swap of neuron (layer, index) between token pos of a and the
// last token of b: each runs with the neuron set to its value in the other.
// Returns for each text the neuron's value, and the logits without and with
// the swap.
export async function neuronSwap(engine, a, pos, b, layer, index) {
  const at = (r) => r.neurons[layer * engine.config.intermediate_size + index];
  const last = b.length - 1;
  const baseB = await runTo(engine, b, last);
  const baseA = await runTo(engine, a, pos);
  const swappedA = await runSet(engine, a, pos, layer, index, at(baseB));
  await runTo(engine, b, last);
  const swappedB = await runSet(engine, b, last, layer, index, at(baseA));
  return {
    a: { value: at(baseA), logits: baseA.logits, swapped: swappedA.logits },
    b: { value: at(baseB), logits: baseB.logits, swapped: swappedB.logits },
  };
}
