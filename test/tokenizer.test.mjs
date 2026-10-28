// The tokenizer against the Hugging Face tokenizer (make_tokens.py). It
// fetches tokenizer.json from the model's repository, as the page does.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { MODEL_URL } from "../web/model.mjs";
import { createTokenizer } from "../web/tokenizer.mjs";

const cases = JSON.parse(readFileSync(new URL("fixture/tokens.json", import.meta.url)));
const tok = createTokenizer(await (await fetch(MODEL_URL + "tokenizer.json")).json());

for (const { text, ids } of cases) {
  test(`encode ${JSON.stringify(text.slice(0, 30))}`, () => {
    assert.deepEqual(tok.encode(text), ids);
    assert.equal(tok.decode(ids), text);
  });
}
