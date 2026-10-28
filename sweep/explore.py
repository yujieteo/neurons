"""After the sweep: pick the interesting neurons, show them to Claude, and
write the snapshot that the page ships (README, Sweep).

    python sweep/explore.py pack MODEL_DIR SWEEP_DIR
        picks the neurons, recomputes their top contexts exactly, runs the
        steering samples, and writes SWEEP_DIR/explore.json and the label
        packs SWEEP_DIR/packs/*.txt
    python sweep/explore.py snapshot MODEL_DIR SWEEP_DIR OUT_DIR
        joins explore.json with the labels SWEEP_DIR/labels/*.json and writes
        OUT_DIR/explored.json and OUT_DIR/neurons.bin
"""
import collections
import json
import pathlib
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

PER_LAYER = 20          # neurons picked per layer
TOKEN_DETECTORS = 5     # of them, neurons whose top contexts share one or two tokens
EXAMPLES = 8            # contexts shown per neuron
BEFORE, AFTER = 20, 4   # tokens of context around the peak
PROMPTS = ["My favourite thing is", "The history of the city"]
STRENGTHS = [0.5, 1.0]  # steering values, as multiples of the neuron's largest value
NEW_TOKENS = 24
PACK = 50               # neurons per label pack


def load(model_dir):
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).cuda().eval()
    return tok, model


def piece(tok, i):
    return tok.decode([int(i)])


def top_tokens(sweep):
    """The 3 most frequent peak tokens of each neuron's top contexts, [L, F, 3]."""
    peak = sweep["tokens"][sweep["top_i"]]                       # [L, K, F]
    L, K, F = peak.shape
    out = np.zeros((L, F, 3), dtype=np.int64)
    for l in range(L):
        for f in range(F):
            common = collections.Counter(peak[l, :, f].tolist()).most_common(3)
            ids = [c[0] for c in common] + [common[0][0]] * (3 - len(common))
            out[l, f] = ids
    return out


def pick(sweep):
    top_v, mean, std = sweep["top_v"], sweep["mean"], sweep["std"]
    peak = sweep["tokens"][sweep["top_i"][:, :16]]                 # [L, 16, F]
    distinct = np.array([[len(set(peak[l, :, f].tolist())) for f in range(peak.shape[2])]
                         for l in range(peak.shape[0])])
    score = (top_v[:, 0] - mean) / (std + 1e-6)
    chosen = []
    for l in range(score.shape[0]):
        order = np.argsort(-score[l])
        concept = [f for f in order if distinct[l, f] >= 3][:PER_LAYER - TOKEN_DETECTORS]
        token = [f for f in order if distinct[l, f] < 3][:TOKEN_DETECTORS]
        chosen += [(l, int(f)) for f in concept + token]
    return chosen, score, distinct


def neuron_values(model, windows, layer, index):
    """The value of neuron (layer, index) at every position of each window."""
    seen = {}

    def pre(_, args):
        seen["h"] = args[0][..., index].clone()

    h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)
    with torch.no_grad():
        model(torch.tensor(windows, device="cuda"), use_cache=False)
    h.remove()
    return seen["h"].cpu().numpy()


def contexts(tok, model, sweep, layer, index):
    W = int(sweep["window"])
    starts, peaks, used = [], [], set()
    for i in sweep["top_i"][layer, :, index].tolist():
        w = i // W
        if w in used:
            continue
        used.add(w)
        starts.append(w * W)
        peaks.append(i - w * W)
        if len(starts) == EXAMPLES:
            break
    tokens = sweep["tokens"]
    windows = [tokens[s:s + W].tolist() for s in starts]
    values = neuron_values(model, windows, layer, index)
    out = []
    for win, vals, p in zip(windows, values, peaks):
        lo, hi = max(0, p - BEFORE), min(W, p + AFTER + 1)
        out.append({"tokens": [piece(tok, t) for t in win[lo:hi]],
                    "values": [round(float(v), 2) for v in vals[lo:hi]], "peak": p - lo})
    return out


def generate(tok, model, steer):
    """Greedy continuations of PROMPTS for each (layer, index, value) in steer,
    with that neuron set to value at every position, one row per pair."""
    rows = [(s, p) for s in steer for p in PROMPTS]
    by_layer = collections.defaultdict(list)
    for r, ((l, f, v), _) in enumerate(rows):
        by_layer[l].append((r, f, v))
    hooks = []
    for l, items in by_layer.items():
        rr = torch.tensor([r for r, _, _ in items], device="cuda")
        ff = torch.tensor([f for _, f, _ in items], device="cuda")
        vv = torch.tensor([v for _, _, v in items], device="cuda")

        def pre(_, args, rr=rr, ff=ff, vv=vv):
            h = args[0].clone()
            h[rr, :, ff] = vv[:, None]
            return (h,)
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
    # The prompts are left-padded to one length; the pad is masked out.
    enc = [tok.encode(p).ids for _, p in rows]
    n = max(map(len, enc))
    ids = torch.tensor([[0] * (n - len(e)) + e for e in enc], device="cuda")
    mask = torch.tensor([[0] * (n - len(e)) + [1] * len(e) for e in enc], device="cuda")
    with torch.no_grad():
        out = model.generate(ids, attention_mask=mask, max_new_tokens=NEW_TOKENS, do_sample=False,
                             pad_token_id=0, eos_token_id=None)
    for h in hooks:
        h.remove()
    return [tok.decode(o[n:].tolist()) for o in out]


def pack(model_dir, sweep_dir):
    tok, model = load(model_dir)
    sweep = dict(np.load(sweep_dir / "sweep.npz"))
    chosen, score, distinct = pick(sweep)
    neurons = []
    for n, (l, f) in enumerate(chosen):
        neurons.append({"layer": l, "index": f, "max": round(float(sweep["top_v"][l, 0, f]), 2),
                        "mean": round(float(sweep["mean"][l, f]), 4), "std": round(float(sweep["std"][l, f]), 4),
                        "score": round(float(score[l, f]), 1), "distinct": int(distinct[l, f]),
                        "examples": contexts(tok, model, sweep, l, f)})
        if n % 100 == 0:
            print(f"contexts {n}/{len(chosen)}", flush=True)
    steer = [(x["layer"], x["index"], k * x["max"]) for x in neurons for k in STRENGTHS]
    texts = []
    for i in range(0, len(steer), 64):
        texts += generate(tok, model, steer[i:i + 64])
    baseline = plain(tok, model)
    per = len(PROMPTS) * len(STRENGTHS)
    for n, x in enumerate(neurons):
        t = texts[n * per:(n + 1) * per]
        x["steer"] = [{"strength": k, "prompt": p, "text": t[a * len(PROMPTS) + b]}
                      for a, k in enumerate(STRENGTHS) for b, p in enumerate(PROMPTS)]
    out = {"prompts": PROMPTS, "baseline": baseline, "strengths": STRENGTHS, "neurons": neurons}
    (sweep_dir / "explore.json").write_text(json.dumps(out, ensure_ascii=False))
    packs = sweep_dir / "packs"
    packs.mkdir(exist_ok=True)
    for p in range(0, len(neurons), PACK):
        (packs / f"pack_{p // PACK:02d}.txt").write_text(render(neurons[p:p + PACK], PROMPTS, baseline))


def plain(tok, model):
    enc = [tok.encode(p).ids for p in PROMPTS]
    out = []
    with torch.no_grad():
        for e in enc:
            g = model.generate(torch.tensor([e], device="cuda"), max_new_tokens=NEW_TOKENS, do_sample=False,
                               pad_token_id=0, eos_token_id=None)
            out.append(tok.decode(g[0, len(e):].tolist()))
    return out


def render(neurons, prompts, baseline):
    """A label pack: each neuron's contexts with the peak token in [[ ]],
    and its steering samples."""
    lines = ["Unsteered continuations:"]
    lines += [f"  {p!r} -> {b!r}" for p, b in zip(prompts, baseline)]
    for x in neurons:
        lines += ["", f"=== L{x['layer']}.N{x['index']}  max {x['max']}  mean {x['mean']}  std {x['std']}"]
        for e in x["examples"]:
            t = list(e["tokens"])
            t[e["peak"]] = f"[[{t[e['peak']]}]]"
            lines.append(f"  {e['values'][e['peak']]:6.2f}  {''.join(t)!r}")
        for s in x["steer"]:
            lines.append(f"  steer x{s['strength']}: {s['prompt']!r} -> {s['text']!r}")
    return "\n".join(lines) + "\n"


def snapshot(model_dir, sweep_dir, out_dir):
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    sweep = dict(np.load(sweep_dir / "sweep.npz"))
    explore = json.loads((sweep_dir / "explore.json").read_text())
    labels = {}
    for f in sorted((sweep_dir / "labels").glob("*.json")):
        for x in json.loads(f.read_text()):
            labels[(x["layer"], x["index"])] = x
    neurons = []
    for x in explore["neurons"]:
        lab = labels.get((x["layer"], x["index"]))
        if not lab:
            continue
        neurons.append({"layer": x["layer"], "index": x["index"], "label": lab["label"],
                        "description": lab["description"], "max": x["max"], "examples": x["examples"],
                        "steer": x["steer"]})
    top = top_tokens(sweep)
    L, F = sweep["mean"].shape
    stats = np.stack([sweep["mean"], sweep["std"], sweep["top_v"][:, 0]], axis=-1).astype(np.float32)
    vocab = {int(i): piece(tok, i) for i in np.unique(top)}
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": "HuggingFaceTB/SmolLM2-135M", "layers": L, "neurons_per_layer": F,
            "tokens": int(sweep["tokens"].size), "window": int(sweep["window"]),
            "prompts": explore["prompts"], "baseline": explore["baseline"], "strengths": explore["strengths"]}
    (out_dir / "explored.json").write_text(json.dumps(
        {"meta": meta, "vocab": vocab, "neurons": neurons}, ensure_ascii=False, separators=(",", ":")))
    # neurons.bin: f32 [L][F][mean, std, max], then u16 [L][F][3] top token ids
    with open(out_dir / "neurons.bin", "wb") as f:
        f.write(stats.tobytes())
        f.write(top.astype(np.uint16).tobytes())


if __name__ == "__main__":
    if sys.argv[1] == "pack":
        pack(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    else:
        snapshot(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))
