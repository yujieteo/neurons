"""The full sweep: every MLP neuron of SmolLM2-135M over a text corpus.

For each neuron it keeps the mean, the standard deviation, the largest value
and the TOPK windows where it is largest. A neuron is one element of
silu(gate_proj(x)) * up_proj(x), the input of down_proj, as in the engine.

    python sweep/sweep.py MODEL_DIR OUT_DIR TOKENS CORPUS.parquet[:SHARE] ...

MODEL_DIR has config.json, tokenizer.json and model.safetensors. Each corpus
is a parquet file with a "text" column; SHARE is its part of TOKENS (equal
parts if not given). OUT_DIR gets sweep.npz (README, Sweep).
"""
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

WINDOW = 128      # tokens per window; the documents run on, joined by <|endoftext|>
SKIP = 4          # the first positions of a window, left out of the statistics
TOPK = 32
BATCH = 32
EOT = 0


def windows(tok, path, budget):
    """Token windows of WINDOW tokens from the texts of one parquet file."""
    buf, made = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
        for enc in tok.encode_batch(batch.column(0).to_pylist()):
            buf.extend(enc.ids)
            buf.append(EOT)
            while len(buf) >= WINDOW:
                yield buf[:WINDOW]
                buf = buf[WINDOW:]
                made += WINDOW
                if made >= budget:
                    return


def main():
    model_dir, out_dir, total = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
    corpora = []
    for arg in sys.argv[4:]:
        path, _, share = arg.partition(":")
        corpora.append((path, float(share) if share else 1.0 / len(sys.argv[4:])))
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).cuda().eval()
    cfg = model.config
    L, F = cfg.num_hidden_layers, cfg.intermediate_size

    s1 = torch.zeros(L, F, dtype=torch.float64, device="cuda")
    s2 = torch.zeros(L, F, dtype=torch.float64, device="cuda")
    top_v = torch.full((L, TOPK, F), -1e9, device="cuda")
    top_i = torch.zeros((L, TOPK, F), dtype=torch.int64, device="cuda")
    state = {"base": 0}

    def hook(layer):
        def pre(_, args):
            h = args[0][:, SKIP:, :].float()             # [B, WINDOW - SKIP, F]
            b = h.shape[0]
            flat = h.reshape(-1, F)
            s1[layer] += flat.sum(0, dtype=torch.float64)
            s2[layer] += (flat.double() ** 2).sum(0)
            pos = torch.arange(SKIP, WINDOW, device="cuda")
            idx = (state["base"] + torch.arange(b, device="cuda")[:, None] * WINDOW + pos[None, :]).reshape(-1)
            vals = torch.cat([top_v[layer], flat])
            v, k = vals.topk(TOPK, dim=0)
            ids = torch.cat([top_i[layer], idx[:, None].expand(-1, F)])
            top_v[layer] = v
            top_i[layer] = ids.gather(0, k)
        return pre

    for i, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.register_forward_pre_hook(hook(i))

    all_ids, count, t0 = [], 0, time.time()
    with torch.no_grad():
        for path, share in corpora:
            pending = []
            for w in windows(tok, path, int(total * share)):
                pending.append(w)
                if len(pending) == BATCH:
                    x = torch.tensor(pending, device="cuda")
                    model(x, use_cache=False)
                    state["base"] += x.numel()
                    all_ids.append(np.array(pending, dtype=np.int32))
                    count += x.numel()
                    pending = []
                    if count % (BATCH * WINDOW * 200) == 0:
                        print(f"{count} tokens, {count / (time.time() - t0):.0f} tokens/s", flush=True)
    n = count // WINDOW * (WINDOW - SKIP)
    mean = (s1 / n)
    std = (s2 / n - mean ** 2).clamp(min=0).sqrt()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "sweep.npz", tokens=np.concatenate(all_ids).reshape(-1), window=WINDOW, skip=SKIP,
             mean=mean.float().cpu().numpy(), std=std.float().cpu().numpy(),
             top_v=top_v.cpu().numpy(), top_i=top_i.cpu().numpy())
    print(f"done: {count} tokens in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
