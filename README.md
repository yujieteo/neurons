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

- `test/engine.test.mjs` runs a tiny random Llama model in the wasm engine and compares it with transformers in f32: the logits, the neuron values, the key-value cache and a steered neuron. `test/make_fixture.py` makes the model.
  Node has no WebGPU, so `npm test` runs the CPU path only.
  Deno has WebGPU, so `deno test` runs the CPU path and the GPU path. It also compares the two paths on the prompts of the fixture, unsteered and steered. With WebGPU but no adapter, the GPU tests fail.
  The tolerance between the two paths: for each compared tensor (the logits of a forward call, and its neuron values), max |gpu - cpu| <= 1e-5 x max |cpu|.
  When `NEURONS_MODEL_DIR` names a directory with `config.json`, `tokenizer.json` and `model.safetensors` of SmolLM2-135M at the revision of `MODEL_URL` (`web/model.mjs`), `deno test` also compares the two paths on SmolLM2-135M: 6 prompts, each unsteered and with 2 steering cases. Without it, that test is skipped.
  In Deno, wgpu on an NVIDIA T1000 refuses a buffer of 64 MiB or more, so the SmolLM2-135M test ran on lavapipe (`VK_ICD_FILENAMES` set to the lavapipe ICD).
  Deno needs `libvulkan.so.1` on its library path to find a Vulkan adapter (on NixOS, from `nixpkgs#vulkan-loader`).
- `test/tokenizer.test.mjs` compares the tokenizer with the Hugging Face tokenizer on 9 texts (`test/make_tokens.py`). It downloads `tokenizer.json` from Hugging Face.

CI runs `npm test`, and `deno test` on lavapipe (Mesa's Vulkan on the CPU), because the runner has no GPU.
Thus CI checks the WGSL kernels on one Vulkan implementation. It does not run a real GPU, a browser or SmolLM2-135M.

On SmolLM2-135M, the largest difference from transformers was 3e-5 in the logits and 7e-5 in the neuron values.
In Node, one new token took 52 ms.

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
