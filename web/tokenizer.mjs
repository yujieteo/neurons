// The byte-level BPE tokenizer of SmolLM2 (a tokenizer.json of the Hugging
// Face tokenizers format): added tokens split first, then each digit alone,
// then the GPT-2 split into words, then the merges inside each word.

const WORDS = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

// GPT-2's map from bytes to printable characters, and back.
function byteChars() {
  const keep = [];
  for (let b = 33; b <= 126; b++) keep.push(b);
  for (let b = 161; b <= 172; b++) keep.push(b);
  for (let b = 174; b <= 255; b++) keep.push(b);
  const map = new Array(256);
  let extra = 0;
  for (let b = 0; b < 256; b++) map[b] = String.fromCharCode(keep.includes(b) ? b : 256 + extra++);
  return map;
}

export function createTokenizer(json) {
  const vocab = new Map(Object.entries(json.model.vocab));
  const ranks = new Map(json.model.merges.map((m, i) => [typeof m === "string" ? m : m.join(" "), i]));
  const added = json.added_tokens.map((t) => t.content);
  for (const t of json.added_tokens) vocab.set(t.content, t.id);
  const pieces = [];
  for (const [s, id] of vocab) pieces[id] = s;
  const toChar = byteChars();
  const toByte = new Map(toChar.map((c, b) => [c, b]));
  const encoder = new TextEncoder();
  const cache = new Map();
  const special = added.length
    ? new RegExp(`(${added.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`)
    : null;

  function bpe(word) {
    if (cache.has(word)) return cache.get(word);
    let parts = Array.from(word);
    for (;;) {
      let best = -1, rank = Infinity;
      for (let i = 0; i + 1 < parts.length; i++) {
        const r = ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < rank) { rank = r; best = i; }
      }
      if (best < 0) break;
      const a = parts[best], b = parts[best + 1], merged = [];
      for (let i = 0; i < parts.length; i++) {
        if (i + 1 < parts.length && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i++; }
        else merged.push(parts[i]);
      }
      parts = merged;
    }
    const ids = parts.map((p) => vocab.get(p));
    cache.set(word, ids);
    return ids;
  }

  function encodeText(text, ids) {
    for (const piece of text.split(/(\p{N})/u)) {
      if (!piece) continue;
      for (const [w] of piece.matchAll(WORDS)) {
        let chars = "";
        for (const b of encoder.encode(w)) chars += toChar[b];
        ids.push(...bpe(chars));
      }
    }
  }

  return {
    encode(text) {
      const ids = [];
      const parts = special ? text.split(special) : [text];
      for (let i = 0; i < parts.length; i++) {
        if (i % 2) ids.push(vocab.get(parts[i]));
        else if (parts[i]) encodeText(parts[i], ids);
      }
      return ids;
    },
    // The bytes of one token; a token can hold part of a UTF-8 character.
    bytes(id) {
      const p = pieces[id] ?? "";
      if (added.includes(p)) return encoder.encode(p);
      return Uint8Array.from(Array.from(p), (c) => toByte.get(c));
    },
    decode(ids) {
      const parts = ids.map((id) => this.bytes(id));
      const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
      let k = 0;
      for (const p of parts) { all.set(p, k); k += p.length; }
      return new TextDecoder().decode(all);
    },
    size: pieces.length,
  };
}
