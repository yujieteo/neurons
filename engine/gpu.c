// neurons: the GPU path of the engine, on WebGPU through Emscripten's port
// (emdawnwebgpu). engine/llama.goose checks each call and passes its state
// here through extern fns. The kernels are engine/kernels.wgsl.
//
// The page gives the device to the module as preinitializedWebGPUDevice.
// gpu_init and gpu_forward return at once. The results arrive later: the
// module calls Module.gpuDone(ok) when the GPU has finished the work.

#include <emscripten.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <webgpu/webgpu.h>

static const char kernels[] = {
#embed "kernels.wgsl"
    , 0};

// The slices of engine/llama.goose (packed { data, len }).
#pragma pack(push, 1)
typedef struct { const float *data; int64_t len; } f32s;
typedef struct { const int32_t *data; int64_t len; } i32s;
typedef struct { float *data; int64_t len; } f32w;
#pragma pack(pop)

double offset_of(int32_t kind, int32_t layer);   // export of engine/llama.goose

EM_JS(void, gpu_done, (int ok), { Module["gpuDone"](ok); });

enum { EMBED, RMS_NORM, MATMUL, MATVEC, ROPE, STORE_KV, ATTENTION, NEURONS, N_KERNELS };
static const char *entry[N_KERNELS] = {"embed", "rms_norm", "matmul", "matvec", "rope", "store_kv", "attention", "neurons"};

// The fixed values of a dispatch: Step of kernels.wgsl.
typedef struct {
    uint32_t n_out, n_in, layer, flags;
    float eps, scale;
    uint32_t group, hd;
} Step;

// One dispatch of a forward call: x workgroups, and y from the tokens. A
// matmul has a matvec too (vec, vx), for calls of one token.
typedef struct {
    int kernel;
    WGPUBindGroup group, vec;
    uint32_t x, vx;
    int y;      // 1: one per token; 8: one per 8 tokens; 0: one
} Dispatch;

// 14 dispatches for each of up to MAXL layers, and 3 more. MAXT and MAXSTEER
// are those of engine/llama.goose.
enum { MAX_DISPATCH = 14 * 32 + 3, STEP_STRIDE = 256, MAXT = 128, MAXSTEER = 64 };

static WGPUDevice device;
static WGPUQueue queue;
static WGPUComputePipeline pipe[N_KERNELS];
static WGPUBuffer call_buf, step_buf, tokens_buf, steer_buf, logits_buf, rec_buf, rb_logits, rb_rec;
static Dispatch dispatch[MAX_DISPATCH];
static int n_dispatch;
static Step steps[MAX_DISPATCH];
static int64_t vocab, stride, rec_len;

// The callbacks of the work in flight, and whether all of them succeeded.
static int pending, all_ok;
static float *out_logits, *out_rec;
static int64_t rec_at, rec_bytes;

static void finish(int ok) {
    all_ok &= ok;
    if (--pending == 0) gpu_done(all_ok);
}

static void on_scope(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void *u1, void *u2) {
    int ok = status == WGPUPopErrorScopeStatus_Success && type == WGPUErrorType_NoError;
    if (!ok) fprintf(stderr, "WebGPU error: %.*s\n", (int)message.length, message.data ? message.data : "");
    finish(ok);
}

static void pop_scopes(int n) {
    for (int i = 0; i < n; i++) {
        wgpuDevicePopErrorScope(device, (WGPUPopErrorScopeCallbackInfo){
            .mode = WGPUCallbackMode_AllowSpontaneous, .callback = on_scope});
    }
}

static void on_map(WGPUMapAsyncStatus status, WGPUStringView message, void *buffer, void *u2) {
    int ok = status == WGPUMapAsyncStatus_Success;
    if (ok && buffer == rb_logits) wgpuBufferReadMappedRange(rb_logits, 0, out_logits, vocab * 4);
    if (ok && buffer == rb_rec) wgpuBufferReadMappedRange(rb_rec, 0, out_rec + rec_at, rec_bytes);
    if (ok) wgpuBufferUnmap(buffer);
    finish(ok);
}

static void map(WGPUBuffer b, int64_t bytes) {
    wgpuBufferMapAsync(b, WGPUMapMode_Read, 0, bytes, (WGPUBufferMapCallbackInfo){
        .mode = WGPUCallbackMode_AllowSpontaneous, .callback = on_map, .userdata1 = b});
}

static WGPUBuffer buffer(int64_t bytes, WGPUBufferUsage usage) {
    return wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor){
        .usage = usage, .size = (uint64_t)((bytes + 15) & ~15)});
}

static WGPUBuffer storage(int64_t bytes) {
    return buffer(bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
}

// The empty submit lets the queue release the staging copy of the tensor.
// wgpu (Deno) runs out of memory on all the writes without a submit.
static WGPUBuffer upload(const float *data, int64_t count) {
    WGPUBuffer b = storage(count * 4);
    wgpuQueueWriteBuffer(queue, b, 0, data, count * 4);
    wgpuQueueSubmit(queue, 0, 0);
    return b;
}

static uint32_t blocks(int64_t n, int64_t size) { return (uint32_t)((n + size - 1) / size); }

// A bind group of kernel with the fixed values of dispatch i and buffers
// (binding 2..).
static WGPUBindGroup bind(int kernel, int i, int n, const WGPUBuffer *bufs) {
    WGPUBindGroupEntry e[6] = {
        {.binding = 0, .buffer = call_buf, .size = 16},
        {.binding = 1, .buffer = step_buf, .offset = i * STEP_STRIDE, .size = sizeof(Step)},
    };
    for (int k = 0; k < n; k++) e[2 + k] = (WGPUBindGroupEntry){.binding = 2 + k, .buffer = bufs[k], .size = WGPU_WHOLE_SIZE};
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipe[kernel], 0);
    WGPUBindGroup g = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
        .layout = layout, .entryCount = 2 + n, .entries = e});
    wgpuBindGroupLayoutRelease(layout);
    return g;
}

// Adds a dispatch of kernel with its fixed values and buffers.
static void add(int kernel, Step s, uint32_t x, int y, int n, const WGPUBuffer *bufs) {
    Dispatch d = {kernel, bind(kernel, n_dispatch, n, bufs), 0, x, 0, y};
    if (kernel == MATMUL) d.vec = bind(MATVEC, n_dispatch, n, bufs), d.vx = blocks(s.n_out, 4);
    steps[n_dispatch] = s;
    dispatch[n_dispatch++] = d;
}

// Copies the weights w (layout of offset_of), the rotary tables and the
// shape to the GPU, and records the dispatches of a forward call.
// Gives Module.gpuDone(ok) when the GPU has them.
int32_t gpu_init(f32s w, f32s rope_cos, f32s rope_sin, int64_t n_vocab, int64_t dim, int64_t ffn,
                 int64_t n_layers, int64_t n_heads, int64_t n_kv, float eps, int64_t max_ctx, int64_t max_rec) {
    int64_t hd = dim / n_heads, kvd = n_kv * hd;
    if (max_ctx > 256) return 0;   // scores of the attention kernel
    if (!device) {
        device = emscripten_webgpu_get_device();
        queue = wgpuDeviceGetQueue(device);
    }
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_OutOfMemory);
    WGPUShaderSourceWGSL wgsl = {.chain = {.sType = WGPUSType_ShaderSourceWGSL},
                                 .code = {kernels, WGPU_STRLEN}};
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &(WGPUShaderModuleDescriptor){
        .nextInChain = &wgsl.chain});
    for (int k = 0; k < N_KERNELS; k++) {
        pipe[k] = wgpuDeviceCreateComputePipeline(device, &(WGPUComputePipelineDescriptor){
            .compute = {.module = module, .entryPoint = {entry[k], WGPU_STRLEN}}});
    }
    wgpuShaderModuleRelease(module);

    vocab = n_vocab;
    stride = n_layers * ffn;
    rec_len = max_rec < max_ctx * stride ? max_rec : max_ctx * stride;
    WGPUBuffer wb[11][32];
    for (int kind = 0; kind <= 10; kind++) {
        for (int l = 0; l < (kind == 0 || kind == 10 ? 1 : n_layers); l++) {
            int64_t at = (int64_t)offset_of(kind, l);
            int64_t end = kind == 0 ? (int64_t)offset_of(1, 0)
                        : kind == 10 ? at + dim
                        : kind == 9 ? (int64_t)offset_of(l + 1 < n_layers ? 1 : 10, l + 1 < n_layers ? l + 1 : 0)
                        : (int64_t)offset_of(kind + 1, l);
            wb[kind][l] = upload(w.data + at, end - at);
        }
    }
    WGPUBuffer cosb = upload(rope_cos.data, max_ctx * hd / 2), sinb = upload(rope_sin.data, max_ctx * hd / 2);
    WGPUBuffer x = storage(MAXT * dim * 4), xn = storage(MAXT * dim * 4), q = storage(MAXT * dim * 4),
               k = storage(MAXT * kvd * 4), v = storage(MAXT * kvd * 4), att = storage(MAXT * dim * 4),
               gate = storage(MAXT * ffn * 4), up = storage(MAXT * ffn * 4);
    call_buf = buffer(16, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    step_buf = buffer(MAX_DISPATCH * STEP_STRIDE, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    tokens_buf = storage(MAXT * 4);
    steer_buf = storage(MAXSTEER * 12);
    logits_buf = storage(n_vocab * 4);
    rec_buf = storage(rec_len * 4);
    rb_logits = buffer(n_vocab * 4, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    rb_rec = buffer(rec_len * 4, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);

    // The dispatches of forward in engine/llama.goose, in the same order.
    n_dispatch = 0;
    Step s = {.n_out = dim, .eps = eps};
    add(EMBED, s, blocks(dim, 64), 1, 3, (WGPUBuffer[]){tokens_buf, wb[0][0], x});
    for (int l = 0; l < n_layers; l++) {
        WGPUBuffer kc = storage(max_ctx * kvd * 4), vc = storage(max_ctx * kvd * 4);
        Step mm = {.n_out = dim, .n_in = dim}, mk = {.n_out = kvd, .n_in = dim};
        add(RMS_NORM, s, 1, 1, 3, (WGPUBuffer[]){x, wb[1][l], xn});
        add(MATMUL, mm, blocks(dim, 64), 8, 3, (WGPUBuffer[]){xn, wb[2][l], q});
        add(MATMUL, mk, blocks(kvd, 64), 8, 3, (WGPUBuffer[]){xn, wb[3][l], k});
        add(MATMUL, mk, blocks(kvd, 64), 8, 3, (WGPUBuffer[]){xn, wb[4][l], v});
        add(ROPE, (Step){.n_out = n_heads, .hd = hd}, blocks(n_heads * hd / 2, 64), 1, 3, (WGPUBuffer[]){cosb, sinb, q});
        add(ROPE, (Step){.n_out = n_kv, .hd = hd}, blocks(kvd / 2, 64), 1, 3, (WGPUBuffer[]){cosb, sinb, k});
        add(STORE_KV, (Step){.n_out = kvd}, blocks(kvd, 64), 1, 4, (WGPUBuffer[]){k, v, kc, vc});
        add(ATTENTION, (Step){.n_out = dim, .scale = 1.0f / sqrtf((float)hd), .group = n_heads / n_kv, .hd = hd},
            n_heads, 1, 4, (WGPUBuffer[]){q, kc, vc, att});
        add(MATMUL, (Step){.n_out = dim, .n_in = dim, .flags = 1}, blocks(dim, 64), 8, 3, (WGPUBuffer[]){att, wb[5][l], x});
        add(RMS_NORM, s, 1, 1, 3, (WGPUBuffer[]){x, wb[6][l], xn});
        add(MATMUL, (Step){.n_out = ffn, .n_in = dim}, blocks(ffn, 64), 8, 3, (WGPUBuffer[]){xn, wb[7][l], gate});
        add(MATMUL, (Step){.n_out = ffn, .n_in = dim}, blocks(ffn, 64), 8, 3, (WGPUBuffer[]){xn, wb[8][l], up});
        add(NEURONS, (Step){.n_out = ffn, .n_in = n_layers, .layer = l}, blocks(ffn, 64), 1, 4,
            (WGPUBuffer[]){gate, up, steer_buf, rec_buf});
        add(MATMUL, (Step){.n_out = dim, .n_in = ffn, .flags = 1}, blocks(dim, 64), 8, 3, (WGPUBuffer[]){gate, wb[9][l], x});
    }
    add(RMS_NORM, s, 1, 1, 3, (WGPUBuffer[]){x, wb[10][0], xn});
    add(MATMUL, (Step){.n_out = n_vocab, .n_in = dim, .flags = 2}, 0, 0, 3,
        (WGPUBuffer[]){xn, wb[0][0], logits_buf});
    for (int i = 0; i < n_dispatch; i++) wgpuQueueWriteBuffer(queue, step_buf, i * STEP_STRIDE, &steps[i], sizeof(Step));

    pending = 2;
    all_ok = 1;
    pop_scopes(2);
    return 1;
}

// Runs tokens[0..nt] at positions start.. with the steered neurons, as
// forward in engine/llama.goose. Gives Module.gpuDone(ok) when the logits of
// the last token, and the neuron values when rec_on, are in logits and rec.
int32_t gpu_forward(i32s tokens, int64_t nt, int64_t start, i32s steer_layer, i32s steer_index, f32s steer_value,
                    int64_t n_steer, uint8_t rec_on, f32w logits, f32w rec) {
    if (!n_dispatch) return 0;
    uint32_t call[4] = {(uint32_t)nt, (uint32_t)start, (uint32_t)n_steer, rec_on};
    wgpuQueueWriteBuffer(queue, call_buf, 0, call, sizeof call);
    wgpuQueueWriteBuffer(queue, tokens_buf, 0, tokens.data, nt * 4);
    struct { int32_t layer, index; float value; } steer[MAXSTEER];
    for (int64_t i = 0; i < n_steer; i++) steer[i].layer = steer_layer.data[i], steer[i].index = steer_index.data[i], steer[i].value = steer_value.data[i];
    if (n_steer) wgpuQueueWriteBuffer(queue, steer_buf, 0, steer, n_steer * 12);

    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, 0);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, 0);
    for (int i = 0; i < n_dispatch; i++) {
        Dispatch *d = &dispatch[i];
        int vec = d->vec && (nt == 1 || !d->y);
        wgpuComputePassEncoderSetPipeline(pass, pipe[vec ? MATVEC : d->kernel]);
        wgpuComputePassEncoderSetBindGroup(pass, 0, vec ? d->vec : d->group, 0, 0);
        wgpuComputePassEncoderDispatchWorkgroups(pass, vec ? d->vx : d->x, vec || !d->y ? 1 : blocks(nt, d->y), 1);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderCopyBufferToBuffer(enc, logits_buf, 0, rb_logits, 0, vocab * 4);
    // The rows of the recorded positions start..start + nt that fit.
    int64_t end = (start + nt) * stride < rec_len ? (start + nt) * stride : rec_len;
    rec_at = start * stride;
    rec_bytes = rec_on && end > rec_at ? (end - rec_at) * 4 : 0;
    if (rec_bytes) wgpuCommandEncoderCopyBufferToBuffer(enc, rec_buf, rec_at * 4, rb_rec, 0, rec_bytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, 0);
    wgpuCommandEncoderRelease(enc);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);

    out_logits = logits.data;
    out_rec = rec.data;
    pending = rec_bytes ? 3 : 2;
    all_ok = 1;
    pop_scopes(1);
    map(rb_logits, vocab * 4);
    if (rec_bytes) map(rb_rec, rec_bytes);
    return 1;
}
