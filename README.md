# neurons

A neuron explorer for SmolLM2-135M that runs in the web browser, written in Goose and compiled to WebAssembly.
The model runs on the GPU with WebGPU when the browser has a WebGPU adapter, and on the CPU in WebAssembly when it does not.
If the GPU cannot take the weights, the model runs on the CPU.
The page shows which path runs.

Inspired by [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).

The page shows the MLP neurons of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).
A sweep on a local GPU read all 46,080 neurons over 10 million tokens.
Claude labelled the neurons that stand out most in each layer.
The page ships that work as a snapshot, and it runs the model live:

- Run your own text, and see the strongest neurons at each token.
- Open a neuron, and see its top contexts, its label and its steering samples.
- Steer a neuron: set it to a value, and compare the text that the model writes with and without it.
- Compute the Jacobian lens of a neuron at a token of your text: the next tokens and the later neurons that it moves most.
- Swap a neuron between your text and another text, and compare the next-token probabilities.
- Ask Claude about a neuron with your own OpenRouter key.

## Run it

```sh
python3 -m http.server -d web 8000
# then open http://localhost:8000/
```

The repository has the built `web/engine.wasm` and the snapshot, so the page needs no build step.

## What the page downloads

| What | From | Size |
|---|---|---|
| `config.json`, `tokenizer.json`, `model.safetensors` | `huggingface.co/HuggingFaceTB/SmolLM2-135M`, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (`MODEL_URL` in `web/model.mjs`) | 269 MB, and 2 MB for the tokenizer |
| The page, the wasm and the snapshot | the server of the page | about 3 MB |

The page downloads the model only when you click "Load the model".
The worker keeps the files in the Cache API, so a second visit does not download them again.
The weights are bf16 in the file. The worker converts them to f32 in WebAssembly memory (538 MB).
On the GPU path, the worker also copies them to GPU memory (538 MB).

## Ask Claude

This page cannot ask Claude for you, because a static page cannot keep an API key secret.
Thus each visitor puts an [OpenRouter](https://openrouter.ai/keys) key in the page.
The browser sends the key only to `openrouter.ai`, and keeps it in `sessionStorage` of that tab.
The key never goes to the server of the page or into this repository.
Claude gets the neuron's statistics, its top contexts and steering samples, the visitor's text and the last steering result.

## Lens and swap

Both change one neuron at one token, and run the model again from that token.
They use the forward call of the engine and its steered neurons, so they run on the CPU path and on the GPU path.
`web/probes.mjs` has the two.

### Jacobian lens

The Jacobian lens of neuron (l, i) at token t of your text is the derivative, with respect to the value of the neuron at t, of:

- the logits at t (the next-token prediction), and
- the values of the neurons of layers l + 1 and after, at t.

The lens changes the neuron at t only, so its result is one column of the Jacobian of the forward pass at t.
The engine computes it by central differences: (f(a + h) - f(a - h)) / 2h, with a the neuron's value at t.
The step h is the neuron's standard deviation in the sweep.
On SmolLM2-135M, 5 neurons in layers 3 to 29 gave the same lens within 3e-3 of its largest value, for steps from 0.3 to 3 standard deviations.
Smaller steps gave larger errors, because of the f32 rounding of the logits.
The page shows the 8 tokens that the neuron raises most, the 8 that it lowers most, and the 10 later neurons that it moves most.

The idea comes from these sources:

- The logit lens reads the residual stream as next-token logits ([nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)).
- An MLP neuron adds its output vector to the residual stream, and that vector promotes some tokens of the vocabulary ([Geva et al., 2022](https://arxiv.org/abs/2203.14680)).
  That is the direct effect only. The lens also includes the effect through the later layers.
- A gradient gives a linear estimate of the effect of a change of an activation ([Nanda, 2023, attribution patching](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)).

### Neuron swap

The neuron swap of neuron (l, i) between token t of your text and the last token of another text:

1. Runs each text, and reads the value of the neuron at its token: a in your text, b in the other text.
2. Runs your text again with the neuron set to b at t only, and the other text with the neuron set to a at its last token only.
3. Shows the next-token probabilities of each text before and after the swap.

This is an interchange intervention on one neuron ([Geiger et al., 2021](https://arxiv.org/abs/2106.02997)), the same as activation patching ([Vig et al., 2020](https://arxiv.org/abs/2004.12265); [Meng et al., 2022](https://arxiv.org/abs/2202.05262)) in the two directions.
Steering sets a neuron at every token. The swap sets it at one token only, to a value that the model gave on real text.

The definitions in this section come from the sources above and from the design of this repository.

## How it works

A neuron is one element of `silu(gate_proj(x)) * up_proj(x)`, the input of `down_proj`, in each of the 30 layers.
There are 1,536 neurons in each layer.

| Part | File | What it does |
|---|---|---|
| Engine | `engine/llama.goose` | The forward pass of a Llama-family model with tied embeddings: RMS norm, rotary embedding, grouped-query attention with a key-value cache, the SiLU MLP. It records the neuron values and sets steered neurons. |
| GPU path | `engine/gpu.c`, `engine/kernels.wgsl` | The same forward pass on WebGPU, through the WebGPU port of Emscripten (emdawnwebgpu), with WGSL kernels. The Goose engine checks each call and gives its tokens, steered neurons and buffers to `engine/gpu.c`. |
| Build | `build.sh` | Goose to C (`gen/llama.c`, `gen/llama.h`), then Emscripten to `web/engine.mjs` and `web/engine.wasm` from that C and `engine/gpu.c`. |
| Loader | `web/model.mjs` | Streams the safetensors file into WebAssembly memory, copies it to the GPU on the GPU path, and calls the engine. |
| Tokenizer | `web/tokenizer.mjs` | The byte-level BPE of SmolLM2. |
| Lens and swap | `web/probes.mjs` | The Jacobian lens and the neuron swap, on the forward call of the engine. |
| Worker | `web/worker.mjs` | Runs the model off the main thread. It takes the GPU path when `navigator.gpu` gives an adapter that takes the weights, else the CPU path. |
| Page | `web/index.html`, `web/app.js`, `web/art.mjs`, `web/style.css` | The look of [teoyujie.org](https://teoyujie.org): its fonts, its palette and its art of random walks beside the normal density. |
| Sweep | `sweep/sweep.py`, `sweep/explore.py` | The local sweep, the choice of neurons, the label packs for Claude and the snapshot. |
| Snapshot | `web/snapshot/explored.json`, `web/snapshot/neurons.bin` | The explored neurons, and the statistics of all neurons. |

### Memory

WebAssembly has no address-space reservation, so Emscripten allocates each Goose data stack in full.
Thus the engine keeps its state in fixed-size globals, and the page keeps the weights in memory from `malloc`.
The page gives the weights to `forward` as a slice.
The only data-stack use is the zeroed temporary that `goose_init` builds for each global array.
The largest global array is `rec` (23.6 MB), so `build.sh` sets the stack to 32 MiB.

### Limits

- 128 tokens for each forward call, and 256 positions in the key-value cache.
- The page runs the first 128 tokens of your text, and makes 32 tokens when you steer.
- The capacities in `engine/llama.goose` fit SmolLM2-135M. Keep the engine to this model family: it has no upstream.

## Build

The build needs the Goose compiler at commit `a108677d300cb0e8bbb842c35a353abe67a20c6c`, with its standard library, and Emscripten.
On NixOS:

```sh
git clone https://github.com/aardappel/goose ../goose
git -C ../goose checkout a108677d300cb0e8bbb842c35a353abe67a20c6c
( cd ../goose && nix shell nixpkgs#cmake nixpkgs#clang nixpkgs#lld nixpkgs#gnumake --command \
    bash -c 'cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ && cmake --build build -j16' )
GOOSE_STDLIB=$PWD/../goose/stdlib nix shell nixpkgs#emscripten --command env GOOSE=$PWD/../goose/build/goose ./build.sh
```

The build used Emscripten 5.0.6. On the first build, Emscripten downloads its WebGPU port (emdawnwebgpu).
`build.sh` writes `gen/sha256sums`, and the tests check the sums, so a stale C or wasm fails the tests.

## Tests

```sh
nix shell nixpkgs#nodejs --command npm test
nix shell nixpkgs#deno --command deno test --allow-read --allow-env test/engine.test.mjs
```

- `test/engine.test.mjs` runs a tiny random Llama model in the wasm engine and compares it with transformers in f32: the logits, the neuron values, the key-value cache, a steered neuron, the swap, and the Jacobian lens against the Jacobian from autograd (1e-3 of its largest value). `test/make_fixture.py` makes the model.
  Node has no WebGPU, so `npm test` runs the CPU path only.
  Deno has WebGPU, so `deno test` runs the CPU path and the GPU path. It also compares the two paths on the prompts of the fixture, unsteered and steered, and on the lens and the swap. With WebGPU but no adapter, the GPU tests fail.
  The tolerance between the two paths: for each compared tensor (the logits of a forward call, and its neuron values), max |gpu - cpu| <= 1e-5 x max |cpu|.
  The swap gives the logits of forward calls, so the same tolerance applies to them.
  The lens is a difference of two forward calls over 2h, so its tolerance is 1e-5 x max |cpu value| / h, with the logits or the neuron values without the change as the values.
  When `NEURONS_MODEL_DIR` names a directory with `config.json`, `tokenizer.json` and `model.safetensors` of SmolLM2-135M at the revision of `MODEL_URL` (`web/model.mjs`), `deno test` also compares the two paths on SmolLM2-135M: 6 prompts, each unsteered and with 2 steering cases. Without it, that test is skipped.
  In Deno, wgpu on an NVIDIA T1000 refuses a buffer of 64 MiB or more, so the SmolLM2-135M test ran on lavapipe (`VK_ICD_FILENAMES` set to the lavapipe ICD).
  Deno needs `libvulkan.so.1` on its library path to find a Vulkan adapter (on NixOS, from `nixpkgs#vulkan-loader`).
- `test/tokenizer.test.mjs` compares the tokenizer with the Hugging Face tokenizer on 9 texts (`test/make_tokens.py`). It downloads `tokenizer.json` from Hugging Face.

CI runs `npm test`, and `deno test` on lavapipe (Mesa's Vulkan on the CPU), because the runner has no GPU.
Thus CI checks the WGSL kernels on one Vulkan implementation. It does not run a real GPU, a browser or SmolLM2-135M.

On SmolLM2-135M, the largest difference from transformers was 3e-5 in the logits and 7e-5 in the neuron values.
In Node, one new token took 52 ms.
At the tenth token of a text, the lens took 0.3 s and the swap 0.8 s.

The GPU path against the CPU path on SmolLM2-135M, in Chromium 156 on an NVIDIA T1000, over 6 prompts of up to 128 tokens with 3 steering cases each:

- The largest difference was 7e-5 in the logits (the largest logit was 29, so 2.4e-6 relative) and 6e-3 in the neuron values (the largest value was 2,626, so 2.3e-6 relative). Both are in the tolerance of 1e-5 relative.
- The greedy tokens were the same: 288 of 288, 16 after each prompt.
- One new token took 9 ms on the GPU path and 35 ms on the CPU path. A prompt with the neuron values recorded took 77 ms and 755 ms.

## Sweep

The sweep runs on a local GPU with PyTorch and transformers.
The model weights and the full sweep output stay out of the repository.

```sh
python sweep/sweep.py MODEL_DIR OUT 10000000 fineweb.parquet:0.7 cosmo.parquet:0.3
python sweep/explore.py pack MODEL_DIR OUT
# Claude labels the neurons of OUT/packs/*.txt into OUT/labels/*.json
python sweep/explore.py snapshot MODEL_DIR OUT web/snapshot
```

1. `sweep.py` runs the model over 128-token windows of the corpus. For each neuron, it keeps the mean, the standard deviation and the 32 largest values with their positions. It leaves out the first 4 positions of each window.
2. `explore.py pack` picks 20 neurons in each layer by how far their largest value stands above their mean, in units of their standard deviation. Of the 20, 5 are neurons whose top contexts share one or two tokens. Then it computes 8 top contexts for each neuron again, and the steering samples. It writes the label packs.
3. Claude reads each pack and writes a label and a description for each neuron.
4. `explore.py snapshot` writes the snapshot.

The corpus was the first shard of `fineweb-edu-dedup` (70 %) and of `cosmopedia-v2` (30 %) from [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

## Licence

MIT (see `LICENSE`), for the code of this repository.
SmolLM2-135M is Apache-2.0. The repository does not include the weights.
Goose is Apache-2.0. The generated `gen/llama.c` and `web/engine.wasm` contain the Goose runtime, so `web/GOOSE-LICENSE.txt` has the Goose licence.
The fonts in `web/fonts/` are under the SIL Open Font License (the `-OFL.txt` files).
The contexts in the snapshot are short extracts of FineWeb-Edu (ODC-By) and Cosmopedia (Apache-2.0).
