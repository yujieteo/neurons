// neurons: the forward pass of a Llama-family causal language model
// (SmolLM2-135M), for a web page that reads and steers its MLP neurons.
//
// One block per layer:
//   x += o_proj(attention(rope(q_proj(n1)), rope(k_proj(n1)), v_proj(n1)))
//   x += down_proj(silu(gate_proj(n2)) * up_proj(n2))
// with n1 = rms_norm(x) * input_layernorm and n2 = rms_norm(x) *
// post_attention_layernorm. A neuron is one element of
// silu(gate_proj(n2)) * up_proj(n2), the input of down_proj. The output
// head is the embedding matrix (tied weights).
//
// The page keeps the f32 weights in memory of its own (layout in
// offset_of) and passes them to forward as a slice. Everything else is
// fixed-size globals, so the data stack holds only the zeroed temporaries
// that goose_init builds for them.

import math;
import std;

// ---- capacities ----
let MAXD = 1024;            // model width
let MAXF = 4096;            // MLP width (neurons per layer)
let MAXKV = 512;            // key and value width (kv heads x head width)
let MAXL = 32;              // layers
let MAXHD = 128;            // head width
let MAXCTX = 256;           // positions in the key-value cache
let MAXT = 128;             // tokens per forward call
let MAXV = 65536;           // vocabulary
let MAXREC = 128 * 30 * 1536;   // recorded neuron values (positions x layers x neurons)
let MAXSTEER = 64;

// ---- the model ----
var n_vocab: i64 = 0;
var dim: i64 = 0;
var ffn: i64 = 0;
var n_layers: i64 = 0;
var n_heads: i64 = 0;
var n_kv: i64 = 0;
var head_dim: i64 = 0;
var norm_eps: f32 = 0.0;

// ---- the work buffers ----
var tokens: i32[MAXT];
var x: f32[MAXT * MAXD];        // residual stream
var xn: f32[MAXT * MAXD];       // normed
var q: f32[MAXT * MAXD];
var k: f32[MAXT * MAXKV];
var v: f32[MAXT * MAXKV];
var att: f32[MAXT * MAXD];      // attention output, before o_proj
var gate: f32[MAXT * MAXF];     // gate_proj, then the neurons
var up: f32[MAXT * MAXF];
var tmp: f32[MAXT * MAXD];
var scores: f32[MAXCTX];
var kcache: f32[MAXL * MAXCTX * MAXKV];
var vcache: f32[MAXL * MAXCTX * MAXKV];
var rope_cos: f32[MAXCTX * MAXHD / 2];
var rope_sin: f32[MAXCTX * MAXHD / 2];
var logits: f32[MAXV];

// neuron values of the recorded positions, [position][layer][neuron]
var rec: f32[MAXREC];
var rec_on: bool = false;

// steering: neuron (layer, index) is set to value at every position
var steer_layer: i32[MAXSTEER];
var steer_index: i32[MAXSTEER];
var steer_value: f32[MAXSTEER];
var n_steer: i64 = 0;

// ---- weight layout ----
// 0: embed_tokens [vocab, dim], then for each layer 1: input_layernorm [dim],
// 2: q_proj [dim, dim], 3: k_proj [kv, dim], 4: v_proj [kv, dim],
// 5: o_proj [dim, dim], 6: post_attention_layernorm [dim],
// 7: gate_proj [ffn, dim], 8: up_proj [ffn, dim], 9: down_proj [dim, ffn],
// and last 10: norm [dim]. Each matrix is row-major [out, in], as stored.

fn kv_dim() -> i64 { n_kv * head_dim }

fn tensor_size(kind: i64) -> i64 {
    if kind == 0 { return n_vocab * dim; }
    if kind == 1 || kind == 6 || kind == 10 { return dim; }
    if kind == 2 || kind == 5 { return dim * dim; }
    if kind == 3 || kind == 4 { return kv_dim() * dim; }
    return ffn * dim;
}

fn layer_size() -> i64 {
    var s: i64 = 0;
    for kind in 1..10 { s += tensor_size(kind); }
    return s;
}

fn offset(kind: i64, layer: i64) -> i64 {
    if kind == 0 { return 0; }
    let base = tensor_size(0);
    if kind == 10 { return base + n_layers * layer_size(); }
    var o = base + layer * layer_size();
    for kk in 1..kind { o += tensor_size(kk); }
    return o;
}

// Sets the shape. Returns the number of f32 weights, or 0 if the shape
// does not fit the capacities.
export fn init_model(vocab: i32, d: i32, f: i32, layers: i32, heads: i32, kv_heads: i32,
                     theta: f64, eps: f64) -> f64 {
    if vocab < 1 || d < 1 || f < 1 || layers < 1 || heads < 1 || kv_heads < 1 { return 0.0; }
    let hd = d / heads;
    if vocab > MAXV || d > MAXD || f > MAXF || layers > MAXL || hd > MAXHD
        || kv_heads * hd > MAXKV || heads % kv_heads != 0 || hd % 2 != 0 || hd * heads != d
        || MAXT * layers * f > MAXREC { return 0.0; }
    n_vocab = vocab; dim = d; ffn = f; n_layers = layers;
    n_heads = heads; n_kv = kv_heads; head_dim = hd;
    norm_eps = eps as f32;
    // Llama rotary embedding: angle = pos * theta^(-2i/hd), pairs (i, i + hd/2)
    let half = hd / 2;
    for p in 0..MAXCTX {
        for i in 0..half {
            let a = p * pow(theta, -2.0 * i / hd);
            rope_cos[p * half + i] = cos(a) as f32;
            rope_sin[p * half + i] = sin(a) as f32;
        }
    }
    return offset(10, 0) + dim;
}

export fn offset_of(kind: i32, layer: i32) -> f64 { offset(kind, layer) }
export fn buf_tokens() -> const u8[:] { bytes_of(tokens) }
export fn buf_logits() -> const u8[:] { bytes_of(logits) }
export fn buf_record() -> const u8[:] { bytes_of(rec) }
export fn max_tokens() -> i32 { MAXT }
export fn max_context() -> i32 { MAXCTX }

// Turns the recording of neuron values on (1) or off (0).
export fn set_record(on: i32) { rec_on = on != 0; }

export fn clear_steer() { n_steer = 0; }

// Adds a steered neuron. Returns 0 if it is out of range or the list is full.
export fn add_steer(layer: i32, index: i32, value: f64) -> i32 {
    if n_steer >= MAXSTEER || layer < 0 || layer >= n_layers || index < 0 || index >= ffn { return 0; }
    steer_layer[n_steer] = layer;
    steer_index[n_steer] = index;
    steer_value[n_steer] = value as f32;
    n_steer += 1;
    return 1;
}

// ---- kernels ----

checkfree fn dot(a: const f32[:], b: const f32[:]) -> f32 {
    assert(b.len >= a.len);
    var s: f32 = 0.0;
    for i in a.len { s += a[i] * b[i]; }
    return s;
}

// y += a * xs
checkfree fn axpy(y: f32[:], a: f32, xs: const f32[:]) {
    assert(xs.len >= y.len);
    for i in y.len { y[i] += a * xs[i]; }
}

// y[t][o] = sum_i m[o][i] * xs[t][i], for t < nt. m is [n_out, n_in].
fn matmul(y: f32[:], xs: const f32[:], m: const f32[:], nt: i64, n_out: i64, n_in: i64) {
    for o in 0..n_out {
        let row = m[o * n_in..(o + 1) * n_in];
        for t in 0..nt { y[t * n_out + o] = dot(row, xs[t * n_in..(t + 1) * n_in]); }
    }
}

fn rms_norm(out: f32[:], xs: const f32[:], g: const f32[:], nt: i64) {
    for t in 0..nt {
        let xr = xs[t * dim..(t + 1) * dim];
        let r = 1.0 / sqrt(dot(xr, xr) / dim + norm_eps);
        for i in 0..dim { out[t * dim + i] = xr[i] * r * g[i]; }
    }
}

fn rope(vs: f32[:], nt: i64, heads: i64, start: i64) {
    let half = head_dim / 2;
    let width = heads * head_dim;
    for t in 0..nt {
        let p = start + t;
        for h in 0..heads {
            let b = t * width + h * head_dim;
            for i in 0..half {
                let c = rope_cos[p * half + i];
                let s = rope_sin[p * half + i];
                let a0 = vs[b + i];
                let a1 = vs[b + i + half];
                vs[b + i] = a0 * c - a1 * s;
                vs[b + i + half] = a1 * c + a0 * s;
            }
        }
    }
}

fn attention(layer: i64, nt: i64, start: i64) {
    let kvd = kv_dim();
    let cache = layer * MAXCTX * MAXKV;
    for t in 0..nt {
        let c = cache + (start + t) * MAXKV;
        for i in 0..kvd {
            kcache[c + i] = k[t * kvd + i];
            vcache[c + i] = v[t * kvd + i];
        }
    }
    let group = n_heads / n_kv;
    let scale = 1.0 / sqrt(head_dim as f32);
    for t in 0..nt {
        let p = start + t;
        for h in 0..n_heads {
            let qh = q[t * dim + h * head_dim..t * dim + (h + 1) * head_dim];
            let kb = cache + (h / group) * head_dim;
            var mx: f32 = -1.0e30;
            for j in 0..p + 1 {
                let s = dot(qh, kcache[kb + j * MAXKV..kb + j * MAXKV + head_dim]) * scale;
                scores[j] = s;
                mx = max(mx, s);
            }
            var sum: f32 = 0.0;
            for j in 0..p + 1 {
                let e = exp(scores[j] - mx);
                scores[j] = e;
                sum += e;
            }
            let out = att[t * dim + h * head_dim..t * dim + (h + 1) * head_dim];
            fill(out, 0.0);
            for j in 0..p + 1 {
                axpy(out, scores[j] / sum, vcache[kb + j * MAXKV..kb + j * MAXKV + head_dim]);
            }
        }
    }
}

fn mlp_neurons(layer: i64, nt: i64, start: i64) {
    for i in 0..nt * ffn {
        let g = gate[i];
        gate[i] = g / (1.0 + exp(-g)) * up[i];
    }
    for s in 0..n_steer {
        if steer_layer[s] == layer {
            for t in 0..nt { gate[t * ffn + steer_index[s]] = steer_value[s]; }
        }
    }
    if rec_on {
        let stride = n_layers * ffn;
        for t in 0..nt {
            let base = (start + t) * stride + layer * ffn;
            if base + ffn <= MAXREC {
                for j in 0..ffn { rec[base + j] = gate[t * ffn + j]; }
            }
        }
    }
}

// Whether tokens[0..nt] at positions start..start + nt fit the engine.
fn fits(nt: i32, start: i32) -> bool {
    if nt < 1 || nt > MAXT || start < 0 || start + nt > MAXCTX { return false; }
    for t in 0..nt {
        if tokens[t] < 0 || tokens[t] >= n_vocab { return false; }
    }
    return true;
}

// Runs tokens[0..nt] at positions start..start + nt, after the key-value
// cache of the positions before start, with the weights w. Writes the
// logits of the last token, and the neuron values when recording is on.
// Returns 0 if the call does not fit.
export fn forward(w: const f32[:], nt: i32, start: i32) -> i32 {
    if n_layers == 0 || w.len < offset(10, 0) + dim || !fits(nt, start) { return 0; }
    for t in 0..nt {
        let tok = tokens[t] as i64;
        for i in 0..dim { x[t * dim + i] = w[tok * dim + i]; }
    }
    let kvd = kv_dim();
    let nd = nt * dim;
    for l in 0..n_layers {
        rms_norm(xn, x, w[offset(1, l)..], nt);
        matmul(q, xn, w[offset(2, l)..], nt, dim, dim);
        matmul(k, xn, w[offset(3, l)..], nt, kvd, dim);
        matmul(v, xn, w[offset(4, l)..], nt, kvd, dim);
        rope(q, nt, n_heads, start);
        rope(k, nt, n_kv, start);
        attention(l, nt, start);
        matmul(tmp, att, w[offset(5, l)..], nt, dim, dim);
        for i in 0..nd { x[i] += tmp[i]; }
        rms_norm(xn, x, w[offset(6, l)..], nt);
        matmul(gate, xn, w[offset(7, l)..], nt, ffn, dim);
        matmul(up, xn, w[offset(8, l)..], nt, ffn, dim);
        mlp_neurons(l, nt, start);
        matmul(tmp, gate, w[offset(9, l)..], nt, dim, ffn);
        for i in 0..nd { x[i] += tmp[i]; }
    }
    let last = (nt - 1) * dim;
    rms_norm(xn, x[last..last + dim], w[offset(10, 0)..], 1);
    matmul(logits, xn, w, 1, n_vocab, dim);
    return 1;
}

// ---- the GPU path ----
// engine/gpu.c runs forward on WebGPU with the kernels of engine/kernels.wgsl.
// Its calls return at once, and the module calls Module.gpuDone(ok) when the
// GPU has finished.

extern fn gpu_init(w: const f32[:], cos_table: const f32[:], sin_table: const f32[:], vocab: i64, d: i64,
                   f: i64, layers: i64, heads: i64, kv_heads: i64, eps: f32, max_ctx: i64, max_rec: i64) -> i32;
extern fn gpu_forward(toks: const i32[:], nt: i64, start: i64, s_layer: const i32[:], s_index: const i32[:],
                      s_value: const f32[:], n_s: i64, rec_on_: bool, out_logits: f32[:], out_rec: f32[:]) -> i32;

// Copies the weights w to the GPU. Returns 0 if they do not fit the model.
export fn init_gpu(w: const f32[:]) -> i32 {
    if n_layers == 0 || w.len < offset(10, 0) + dim { return 0; }
    return gpu_init(w, rope_cos, rope_sin, n_vocab, dim, ffn, n_layers, n_heads, n_kv, norm_eps, MAXCTX, MAXREC);
}

// forward on the GPU, with the weights of init_gpu. Returns 0 if the call
// does not fit.
export fn forward_gpu(nt: i32, start: i32) -> i32 {
    if !fits(nt, start) { return 0; }
    return gpu_forward(tokens, nt, start, steer_layer, steer_index, steer_value, n_steer, rec_on, logits, rec);
}
