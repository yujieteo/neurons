"""Writes test/fixture/tokens.json: texts and the token ids that the Hugging
Face tokenizer of SmolLM2-135M gives them. test/tokenizer.test.mjs compares.

    python test/make_tokens.py path/to/tokenizer.json
"""
import json
import pathlib
import sys

from tokenizers import Tokenizer

TEXTS = [
    "The capital of France is Paris.",
    "  two spaces, a tab\there and\nnew lines\n\n",
    "In 2024 the price rose 3.75% to $1,299.",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)",
    "naïve café, Straße, 東京, Привет, 🙂 emoji",
    "don't you'll we've I'm they'd it's",
    "<|endoftext|>a<|im_start|>user\nhi<|im_end|>",
    "١٢٣ Ⅻ ½",
    "",
]

tok = Tokenizer.from_file(sys.argv[1])
out = [{"text": t, "ids": tok.encode(t).ids} for t in TEXTS]
path = pathlib.Path(__file__).parent / "fixture" / "tokens.json"
path.write_text("[\n" + ",\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n]\n")
