// The page look against the site faults: the neuron rows take the site type and colour,
// /shared/kowalski.css owns the disabled look, captions have 4.5:1 contrast,
// and the header is the site header. The keyboard of the neuron lists is checked in a browser.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../web/style.css", import.meta.url), "utf8");
const html = readFileSync(new URL("../web/index.html", import.meta.url), "utf8");
const rule = (selector) => css.match(new RegExp(`(?:^|\\n)${selector.replace(/[.*]/g, "\\$&")} \\{([^}]*)\\}`))?.[1] ?? "";

test("neuron rows inherit the font and the colour", () => {
  assert.match(rule(".neuronlist .item"), /font: inherit;.*color: inherit;/s);
});

test("no local disabled look", () => {
  assert.doesNotMatch(css, /:disabled \{/);
});

test("captions are --muted", () => {
  assert.match(rule(".caption"), /color: var\(--muted\);/);
  assert.doesNotMatch(css, /#82827c/i);
});

test("the site header, with no nav", () => {
  const header = html.match(/<header[^>]*>([\s\S]*?)<\/header>/)[1];
  assert.match(header, /<a class="name" href="https:\/\/teoyujie.org\/">Yu Jie Teo<\/a>/);
  assert.match(header, /<span class="tagline">Mechanical Engineer<\/span>/);
  assert.doesNotMatch(header, /<nav/);
});
