// neurons: the WGSL kernels of the GPU path (engine/gpu.c). Each kernel is
// one step of forward in engine/llama.goose, and does the same arithmetic,
// except for the order of the sums.
//
// Every dispatch binds the per-call values (call) and its own fixed values
// (step) at binding 0 and 1, then its buffers from binding 2.

struct Call {
    nt: u32,        // tokens of this call
    start: u32,     // position of the first token
    n_steer: u32,
    rec_on: u32,
}

struct Step {
    n_out: u32,     // matmul: rows of the matrix; rope: heads; others: width
    n_in: u32,      // matmul: columns of the matrix
    layer: u32,
    flags: u32,     // matmul: 1 adds to y, 2 uses only the last token
    eps: f32,       // rms_norm
    scale: f32,     // attention: 1 / sqrt(head width)
    group: u32,     // attention: query heads per key-value head
    hd: u32,        // head width
}

@group(0) @binding(0) var<uniform> call: Call;
@group(0) @binding(1) var<uniform> step: Step;

// ---- embed: x[t] = embed_tokens[tokens[t]] ----

@group(0) @binding(2) var<storage, read> e_tokens: array<i32>;
@group(0) @binding(3) var<storage, read> e_table: array<f32>;
@group(0) @binding(4) var<storage, read_write> e_x: array<f32>;

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) id: vec3u) {
    let d = step.n_out;
    if id.x >= d || id.y >= call.nt { return; }
    e_x[id.y * d + id.x] = e_table[u32(e_tokens[id.y]) * d + id.x];
}

// ---- rms_norm: out[t] = x[t] / rms(x[t]) * g, one workgroup per token ----

@group(0) @binding(2) var<storage, read> n_x: array<f32>;
@group(0) @binding(3) var<storage, read> n_g: array<f32>;
@group(0) @binding(4) var<storage, read_write> n_out: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn rms_norm(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
    if wg.y >= call.nt { return; }
    let d = step.n_out;
    let b = wg.y * d;
    var s = 0.0;
    for (var i = lid; i < d; i += 64u) { s += n_x[b + i] * n_x[b + i]; }
    partial[lid] = s;
    workgroupBarrier();
    for (var w = 32u; w > 0u; w >>= 1u) {
        if lid < w { partial[lid] += partial[lid + w]; }
        workgroupBarrier();
    }
    let r = 1.0 / sqrt(partial[0] / f32(d) + step.eps);
    for (var i = lid; i < d; i += 64u) { n_out[b + i] = n_x[b + i] * r * n_g[i]; }
}

// ---- matmul: y[t][o] (+)= sum_i m[o][i] * x[t][i], m is [n_out, n_in] ----
// A workgroup computes 64 rows for 8 tokens, in tiles of 16 columns.

@group(0) @binding(2) var<storage, read> m_x: array<f32>;
@group(0) @binding(3) var<storage, read> m_m: array<f32>;
@group(0) @binding(4) var<storage, read_write> m_y: array<f32>;

var<workgroup> tile_m: array<f32, 1088>;    // 64 rows x 17 (16 and a pad)
var<workgroup> tile_x: array<f32, 128>;     // 8 tokens x 16

@compute @workgroup_size(64)
fn matmul(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n_out = step.n_out;
    let n_in = step.n_in;
    let t0 = wg.y * 8u;
    let o0 = wg.x * 64u;
    var acc: array<f32, 8>;
    for (var i0 = 0u; i0 < n_in; i0 += 16u) {
        for (var k = lid; k < 1024u; k += 64u) {
            let r = k / 16u;
            let c = k % 16u;
            var v = 0.0;
            if o0 + r < n_out && i0 + c < n_in { v = m_m[(o0 + r) * n_in + i0 + c]; }
            tile_m[r * 17u + c] = v;
        }
        for (var k = lid; k < 128u; k += 64u) {
            let r = k / 16u;
            let c = k % 16u;
            var v = 0.0;
            if t0 + r < call.nt && i0 + c < n_in { v = m_x[(t0 + r) * n_in + i0 + c]; }
            tile_x[k] = v;
        }
        workgroupBarrier();
        for (var c = 0u; c < 16u; c++) {
            let w = tile_m[lid * 17u + c];
            for (var r = 0u; r < 8u; r++) { acc[r] += w * tile_x[r * 16u + c]; }
        }
        workgroupBarrier();
    }
    let o = o0 + lid;
    if o >= n_out { return; }
    for (var r = 0u; r < 8u; r++) {
        if t0 + r >= call.nt { break; }
        let j = (t0 + r) * n_out + o;
        if (step.flags & 1u) != 0u { m_y[j] += acc[r]; } else { m_y[j] = acc[r]; }
    }
}

// ---- matvec: matmul for one token, the only one or (flags 2) the last ----
// A workgroup computes 4 rows, with 64 threads on each row. The last-token
// matvec writes row 0 of y.

var<workgroup> rows: array<array<f32, 64>, 4>;

@compute @workgroup_size(64, 4)
fn matvec(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let n_in = step.n_in;
    let o = wg.x * 4u + lid.y;
    let last = (step.flags & 2u) != 0u;
    let t = select(0u, call.nt - 1u, last);
    var s = 0.0;
    if o < step.n_out {
        for (var i = lid.x; i < n_in; i += 64u) { s += m_m[o * n_in + i] * m_x[t * n_in + i]; }
    }
    rows[lid.y][lid.x] = s;
    workgroupBarrier();
    for (var w = 32u; w > 0u; w >>= 1u) {
        if lid.x < w { rows[lid.y][lid.x] += rows[lid.y][lid.x + w]; }
        workgroupBarrier();
    }
    if lid.x != 0u || o >= step.n_out { return; }
    let j = select(t * step.n_out + o, o, last);
    if (step.flags & 1u) != 0u { m_y[j] += rows[lid.y][0]; } else { m_y[j] = rows[lid.y][0]; }
}

// ---- rope: the rotary embedding of position start + t, in place ----

@group(0) @binding(2) var<storage, read> r_cos: array<f32>;
@group(0) @binding(3) var<storage, read> r_sin: array<f32>;
@group(0) @binding(4) var<storage, read_write> r_v: array<f32>;

@compute @workgroup_size(64)
fn rope(@builtin(global_invocation_id) id: vec3u) {
    let half = step.hd / 2u;
    if id.x >= step.n_out * half || id.y >= call.nt { return; }
    let h = id.x / half;
    let i = id.x % half;
    let b = id.y * step.n_out * step.hd + h * step.hd;
    let p = call.start + id.y;
    let c = r_cos[p * half + i];
    let s = r_sin[p * half + i];
    let a0 = r_v[b + i];
    let a1 = r_v[b + i + half];
    r_v[b + i] = a0 * c - a1 * s;
    r_v[b + i + half] = a1 * c + a0 * s;
}

// ---- store_kv: k and v of each token into the cache of the layer ----
// The cache of a layer is [position][key-value width].

@group(0) @binding(2) var<storage, read> s_k: array<f32>;
@group(0) @binding(3) var<storage, read> s_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> s_kc: array<f32>;
@group(0) @binding(5) var<storage, read_write> s_vc: array<f32>;

@compute @workgroup_size(64)
fn store_kv(@builtin(global_invocation_id) id: vec3u) {
    let kvd = step.n_out;
    if id.x >= kvd || id.y >= call.nt { return; }
    let c = (call.start + id.y) * kvd + id.x;
    s_kc[c] = s_k[id.y * kvd + id.x];
    s_vc[c] = s_v[id.y * kvd + id.x];
}

// ---- attention: one workgroup per (head, token) ----

@group(0) @binding(2) var<storage, read> a_q: array<f32>;
@group(0) @binding(3) var<storage, read> a_kc: array<f32>;
@group(0) @binding(4) var<storage, read> a_vc: array<f32>;
@group(0) @binding(5) var<storage, read_write> a_out: array<f32>;

var<workgroup> scores: array<f32, 256>;   // MAXCTX of engine/llama.goose

@compute @workgroup_size(64)
fn attention(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let h = wg.x;
    let t = wg.y;
    let hd = step.hd;
    let dim = step.n_out;
    let kvd = dim / step.group;
    let p = call.start + t;
    let qb = t * dim + h * hd;
    let kb = (h / step.group) * hd;
    var mx = -1.0e30;
    for (var j = lid; j <= p; j += 64u) {
        var s = 0.0;
        for (var i = 0u; i < hd; i++) { s += a_q[qb + i] * a_kc[j * kvd + kb + i]; }
        s *= step.scale;
        scores[j] = s;
        mx = max(mx, s);
    }
    partial[lid] = mx;
    workgroupBarrier();
    for (var w = 32u; w > 0u; w >>= 1u) {
        if lid < w { partial[lid] = max(partial[lid], partial[lid + w]); }
        workgroupBarrier();
    }
    mx = partial[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = lid; j <= p; j += 64u) {
        let e = exp(scores[j] - mx);
        scores[j] = e;
        sum += e;
    }
    partial[lid] = sum;
    workgroupBarrier();
    for (var w = 32u; w > 0u; w >>= 1u) {
        if lid < w { partial[lid] += partial[lid + w]; }
        workgroupBarrier();
    }
    sum = partial[0];
    for (var i = lid; i < hd; i += 64u) {
        var o = 0.0;
        for (var j = 0u; j <= p; j++) { o += scores[j] / sum * a_vc[j * kvd + kb + i]; }
        a_out[qb + i] = o;
    }
}

// ---- neurons: gate = silu(gate) * up, then the steered neurons, then the
// record [position][layer][neuron] of positions below MAXT ----

struct Steer {
    layer: i32,
    index: i32,
    value: f32,
}

@group(0) @binding(2) var<storage, read_write> u_gate: array<f32>;
@group(0) @binding(3) var<storage, read> u_up: array<f32>;
@group(0) @binding(4) var<storage, read> u_steer: array<Steer>;
@group(0) @binding(5) var<storage, read_write> u_rec: array<f32>;

@compute @workgroup_size(64)
fn neurons(@builtin(global_invocation_id) id: vec3u) {
    let ffn = step.n_out;
    let j = id.x;
    let t = id.y;
    if j >= ffn || t >= call.nt { return; }
    let g = u_gate[t * ffn + j];
    var v = g / (1.0 + exp(-g)) * u_up[t * ffn + j];
    for (var s = 0u; s < call.n_steer; s++) {
        let st = u_steer[s];
        if u32(st.layer) == step.layer && u32(st.index) == j { v = st.value; }
    }
    u_gate[t * ffn + j] = v;
    let stride = step.n_in * ffn;   // n_in: layers
    let base = (call.start + t) * stride + step.layer * ffn;
    if call.rec_on != 0u && base + ffn <= arrayLength(&u_rec) { u_rec[base + j] = v; }
}
