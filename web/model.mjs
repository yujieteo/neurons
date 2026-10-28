// The Goose engine with a model loaded: weights from a safetensors file,
// streamed into WebAssembly memory as f32, and calls that run it.

// The engine's weight layout (offset_of in engine/llama.goose): kind 0 is the
// embedding, 1 to 9 are per layer, 10 is the final norm.
const KINDS = {
  "input_layernorm.weight": 1, "self_attn.q_proj.weight": 2, "self_attn.k_proj.weight": 3,
  "self_attn.v_proj.weight": 4, "self_attn.o_proj.weight": 5, "post_attention_layernorm.weight": 6,
  "mlp.gate_proj.weight": 7, "mlp.up_proj.weight": 8, "mlp.down_proj.weight": 9,
};

export function tensorPlace(name) {
  if (name === "model.embed_tokens.weight") return [0, 0];
  if (name === "model.norm.weight") return [10, 0];
  const m = /^model\.layers\.(\d+)\.(.+)$/.exec(name);
  if (m && KINDS[m[2]]) return [KINDS[m[2]], Number(m[1])];
  return null;
}

// Converts the bytes of `count` elements of `dtype` at src to f32 at dst.
function convert(dtype, src, dst) {
  if (dtype === "BF16") {
    const s = new Uint16Array(src.buffer, src.byteOffset, src.length >> 1);
    for (let i = 0; i < s.length; i++) dst[i] = s[i] << 16;
  } else {
    throw new Error(`weights of type ${dtype} are not supported`);
  }
}

// With a WebGPU device, the engine runs forward on the GPU (engine/gpu.c);
// without one, or if the GPU fails to take the weights, on the CPU in
// WebAssembly.
export async function createEngine(createModule, device = null) {
  const m = await createModule(device ? { preinitializedWebGPUDevice: device } : {});
  m._goose_init(0, 0);
  const lenPtr = m._malloc(8);
  const slicePtr = m._malloc(12);   // sl_f32 of gen/llama.h: packed { float *data; int64_t len; }
  const view = (fn, Type) => {
    const ptr = fn(lenPtr);
    const len = Number(new BigInt64Array(m.HEAPU8.buffer, lenPtr, 1)[0]);
    return new Type(m.HEAPU8.buffer, ptr, len / Type.BYTES_PER_ELEMENT);
  };
  let weights = 0, nWeights = 0, config = null, gpu = device !== null;
  const weightSlice = () => {
    const dv = new DataView(m.HEAPU8.buffer, slicePtr, 12);
    dv.setUint32(0, weights, true);
    dv.setBigInt64(4, BigInt(nWeights), true);
    return slicePtr;
  };
  // Starts a GPU call, and waits for the GPU to finish it. refused is the
  // error if the engine refuses the call.
  const onGpu = (start, refused) => new Promise((resolve, reject) => {
    m.gpuDone = (ok) => (ok ? resolve() : reject(new Error("the GPU failed")));
    if (!start()) reject(new Error(refused));
  });

  return {
    // Reads a safetensors file from a stream of Uint8Array chunks into the
    // engine. config is the model's config.json. onBytes(n) reports progress.
    async load(cfg, chunks, onBytes = () => {}) {
      const n = m._init_model(cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size,
        cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads,
        cfg.rope_theta, cfg.rms_norm_eps);
      if (!n) throw new Error("the model does not fit the engine's capacities");
      if (cfg.tie_word_embeddings === false) throw new Error("the engine needs tied embeddings");
      if (weights) m._free(weights);
      weights = m._malloc(4 * n);
      if (!weights) throw new Error("not enough memory for the weights");
      nWeights = n;
      config = cfg;
      await readSafetensors(chunks, onBytes, (name, dtype, shape, bytes) => {
        const place = tensorPlace(name);
        if (!place) throw new Error(`unknown tensor ${name}`);
        const off = m._offset_of(place[0], place[1]);
        const count = shape.reduce((a, b) => a * b, 1);
        convert(dtype, bytes, new Uint32Array(m.HEAPU8.buffer, weights + 4 * off, count));
      });
      if (gpu) gpu = await onGpu(() => m._init_gpu(weightSlice()), "the GPU path cannot take these weights").then(
        () => true, (e) => { console.warn(`${e.message}: the model runs on the CPU`); return false; });
    },
    get config() { return config; },
    get path() { return gpu ? "gpu" : "cpu"; },
    maxTokens: () => m._max_tokens(),
    maxContext: () => m._max_context(),
    // Runs ids at positions start.. and returns the logits of the last one.
    async forward(ids, start) {
      new Int32Array(m.HEAPU8.buffer, m._buf_tokens(lenPtr), ids.length).set(ids);
      if (gpu) await onGpu(() => m._forward_gpu(ids.length, start), "forward does not fit");
      else if (!m._forward(weightSlice(), ids.length, start)) throw new Error("forward does not fit");
      return view(m._buf_logits, Float32Array).subarray(0, config.vocab_size);
    },
    setRecord: (on) => m._set_record(on ? 1 : 0),
    // The neuron values at position pos, [layer][neuron], from the last
    // forward calls with recording on.
    neuronsAt(pos) {
      const stride = config.num_hidden_layers * config.intermediate_size;
      return view(m._buf_record, Float32Array).slice(pos * stride, (pos + 1) * stride);
    },
    // steer: a list of { layer, index, value }
    setSteer(steer) {
      m._clear_steer();
      for (const s of steer) if (!m._add_steer(s.layer, s.index, s.value)) throw new Error("cannot steer that neuron");
    },
  };
}

// Calls onTensor(name, dtype, shape, bytes) for each tensor of a safetensors
// stream, in file order, with the bytes copied out of the chunks.
export async function readSafetensors(chunks, onBytes, onTensor) {
  // The chunks not yet used, and their total length. take(n) joins only the
  // n bytes it returns, so each byte is copied once.
  const queue = [];
  let queued = 0, seen = 0, pos = 0, tensors = null, next = 0;
  const take = (n) => {
    const out = new Uint8Array(n);
    for (let k = 0; k < n;) {
      const c = queue[0], m = Math.min(c.length, n - k);
      out.set(c.subarray(0, m), k);
      k += m;
      if (m === c.length) queue.shift(); else queue[0] = c.subarray(m);
    }
    queued -= n; pos += n;
    return out;
  };
  for await (const chunk of chunks) {
    seen += chunk.length;
    onBytes(seen);
    queue.push(chunk);
    queued += chunk.length;
    if (!tensors) {
      if (queued < 8) continue;
      if (queue[0].length < 8) { const all = take(queued); queue.push(all); queued = all.length; }
      const n = Number(new DataView(queue[0].buffer, queue[0].byteOffset, 8).getBigUint64(0, true));
      if (queued < 8 + n) continue;
      take(8);
      const header = JSON.parse(new TextDecoder().decode(take(n)));
      delete header.__metadata__;
      tensors = Object.entries(header).sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);
      pos = 0;
    }
    while (next < tensors.length) {
      const [name, t] = tensors[next];
      const [s, e] = t.data_offsets;
      if (pos < s) { take(Math.min(s - pos, queued)); if (pos < s) break; }
      if (queued < e - s) break;
      onTensor(name, t.dtype, t.shape, take(e - s));
      next++;
    }
  }
  if (!tensors || next < tensors.length) throw new Error("the weights file ended early");
}

// The model the page runs: SmolLM2-135M (Apache-2.0) at a fixed revision of
// its Hugging Face repository.
export const MODEL_URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/";
