"""Writes test/fixture/: a tiny random Llama model (bf16 safetensors) and the
values that transformers computes with it in f32. test/engine.test.mjs runs
the same model in the wasm engine and compares.

    python test/make_fixture.py   (needs torch, transformers, safetensors)
"""
import json
import pathlib

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

OUT = pathlib.Path(__file__).parent / "fixture"
PROMPT = [1, 17, 300, 42, 42, 7, 511, 128, 64, 3]
MORE = [99, 5, 250]
STEER = {"layer": 1, "index": 5, "value": 3.0}
LENS = {"pos": 6, "layer": 0, "index": 5}
SWAP = {"other": [3, 88, 400, 12, 7], "pos": 6, "layer": 0, "index": 11}


def neurons(model, ids, steer=None, past=None):
    """The logits of the last token, the neuron values [pos][layer][neuron]
    and the cache, with neuron steer set to its value at every position, or
    only at the last position if steer has "last". The value may be a tensor."""
    seen = []

    def hook(layer):
        def pre(_, args):
            h = args[0]
            if steer and steer["layer"] == layer:
                h = h.clone()
                h[..., -1 if steer.get("last") else slice(None), steer["index"]] = steer["value"]
            seen.append(h[0].clone())
            return (h,)
        return pre

    hooks = [m.mlp.down_proj.register_forward_pre_hook(hook(i)) for i, m in enumerate(model.model.layers)]
    out = model(torch.tensor([ids]), past_key_values=past, use_cache=True)
    for h in hooks:
        h.remove()
    acts = torch.stack(seen, dim=1)  # [pos, layer, neuron]
    return out.logits[0, -1], acts, out.past_key_values


def lens(model):
    """The Jacobian of the logits and of the neuron values at LENS["pos"]
    with respect to neuron LENS at that position, by autograd."""
    ids = PROMPT[:LENS["pos"] + 1]
    _, acts, _ = neurons(model, ids)
    value = acts[-1, LENS["layer"], LENS["index"]]

    def f(v):
        logits, acts, _ = neurons(model, ids, steer={**LENS, "value": v, "last": True})
        return torch.cat([logits, acts[-1].flatten()])

    jac = torch.autograd.functional.jacobian(f, value)
    return jac[:model.config.vocab_size], jac[model.config.vocab_size:]


def swap(model):
    """The logits of PROMPT up to SWAP["pos"] and of SWAP["other"], each with
    neuron SWAP at its last token set to its value in the other."""
    a, b = PROMPT[:SWAP["pos"] + 1], SWAP["other"]
    _, acts_a, _ = neurons(model, a)
    _, acts_b, _ = neurons(model, b)
    va, vb = (x[-1, SWAP["layer"], SWAP["index"]].item() for x in (acts_a, acts_b))
    logits_a, _, _ = neurons(model, a, steer={**SWAP, "value": vb, "last": True})
    logits_b, _, _ = neurons(model, b, steer={**SWAP, "value": va, "last": True})
    return logits_a, logits_b


def rounded(t):
    return [float(f"{v:.7g}") for v in t.flatten().tolist()]


def main():
    torch.manual_seed(0)
    config = {"vocab_size": 512, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "rope_theta": 10000.0,
              "rms_norm_eps": 1e-5, "tie_word_embeddings": True}
    cfg = LlamaConfig(**config, initializer_range=0.2, max_position_embeddings=256)
    assert cfg.rope_parameters["rope_theta"] == config["rope_theta"]
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)  # the norms start at 1; vary them too
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
        logits, acts, past = neurons(model, PROMPT)
        more_logits, more_acts, _ = neurons(model, MORE, past=past)
        steer_logits, _, _ = neurons(model, PROMPT, steer=STEER)
        swap_a, swap_b = swap(model)
    lens_logits, lens_neurons = lens(model)
    OUT.mkdir(exist_ok=True)
    state = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, OUT / "tiny.safetensors", metadata={"format": "pt"})
    expect = {"config": config, "prompt": PROMPT, "more": MORE, "steer": STEER,
              "logits": rounded(logits), "neurons": rounded(acts),
              "more_logits": rounded(more_logits), "more_neurons": rounded(more_acts),
              "steer_logits": rounded(steer_logits),
              "lens": LENS, "lens_logits": rounded(lens_logits), "lens_neurons": rounded(lens_neurons),
              "swap": SWAP, "swap_a": rounded(swap_a), "swap_b": rounded(swap_b)}
    (OUT / "tiny.json").write_text(json.dumps(expect) + "\n")


if __name__ == "__main__":
    main()
