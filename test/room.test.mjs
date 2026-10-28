// The room client (web/room.mjs) against a fake relay: the bodies that it
// sends, the key check, and the order of views and notes.
import test from "node:test";
import assert from "node:assert/strict";
import { body, joinRoom, KeyError, newId, roomOfHash } from "../web/room.mjs";

const KEY = "sk-or-v1-0123456789abcdef0123456789abcdef";
const ROOM = "AAAAAAAAAAAAAAAAAAAAAA";

// A relay that keeps the requests and gives each one the next reply of replies,
// else an empty state. The reply to the first request waits for release().
function relay(replies = []) {
  const sent = [];
  let release;
  const first = new Promise((r) => { release = r; });
  const fetch = async (url, init) => {
    sent.push(JSON.parse(init.body));
    if (sent.length === 1) await first;
    const r = replies.shift() ?? { status: 200, data: { seq: 1, expires: 0, members: [] } };
    return { ok: r.status === 200, status: r.status, json: async () => r.data };
  };
  return { sent, fetch, release };
}

const until = async (f) => { for (let i = 0; i < 200 && !f(); i++) await new Promise((r) => setTimeout(r, 5)); };

test("ids", () => {
  assert.match(newId(16), /^[A-Za-z0-9_-]{22}$/);
  assert.match(newId(9), /^[A-Za-z0-9_-]{12}$/);
  assert.equal(roomOfHash(`#room=${ROOM}`), ROOM);
  assert.equal(roomOfHash("#16.1275"), null);
});

test("a body has only the fields of its operation", () => {
  const b = JSON.parse(body("note", ROOM, "member01", "Ada", 3, { neuron: "1.2", text: "hi", base: 0, key: KEY }, ""));
  assert.deepEqual(Object.keys(b).sort(), ["base", "member", "name", "neuron", "op", "room", "since", "text"]);
  assert.deepEqual(Object.keys(JSON.parse(body("poll", ROOM, "member01", "Ada", 3, { key: KEY, view: "{}" }, ""))).sort(),
    ["member", "name", "op", "room", "since"]);
});

test("a body cannot carry the OpenRouter key", () => {
  // The visitor's own key, in any field, and any text like an OpenRouter key.
  const own = "my-own-key-0123456789";
  assert.throws(() => body("note", ROOM, "member01", "Ada", 3, { neuron: "1.2", text: `key ${own}`, base: 0 }, own), KeyError);
  assert.throws(() => body("view", ROOM, "member01", "Ada", 3, { view: JSON.stringify({ text: KEY }) }, ""), KeyError);
  assert.throws(() => body("poll", ROOM, "member01", "SK-OR-v1-x", 3, {}, ""), KeyError);
});

test("the room never sends the key from a view or a note", async () => {
  const { sent, fetch, release } = relay();
  release();
  const errors = [];
  const r = joinRoom({ url: "room", room: ROOM, member: "member01", name: () => "Ada", secret: () => KEY,
    onState() {}, onConflict() {}, onError: (e) => errors.push(e), fetch, interval: 10 });
  r.setView({ layer: 1, index: 2, text: `paste ${KEY}`, token: 0, lens: false });
  r.note("1.2", `my key is ${KEY}`, 0);
  r.note("1.3", "fine", 0);
  await until(() => sent.some((m) => m.op === "note"));
  r.leave(() => {});
  assert.ok(sent.length > 0);
  for (const m of sent) assert.ok(!JSON.stringify(m).includes(KEY));
  assert.deepEqual(sent.filter((m) => m.op === "note").map((m) => m.neuron), ["1.3"]);
  assert.deepEqual(errors.map((e) => e.op), ["view", "note"]);
});

test("a view that waits is newer than the view in a reply", async () => {
  const { sent, fetch, release } = relay();
  let r;
  const pending = [];
  r = joinRoom({ url: "room", room: ROOM, member: "member01", name: () => "Ada", secret: () => "",
    onState() { pending.push(r.viewPending()); }, onConflict() {}, onError() {}, fetch, interval: 10 });
  await until(() => sent.length);
  r.setView({ layer: 1 });
  assert.equal(r.viewPending(), true);
  release();
  await until(() => sent.some((m) => m.op === "view"));
  r.leave(() => {});
  // The first request was in flight, so its reply came while the view waited.
  assert.equal(sent[0].op, "poll");
  assert.equal(pending[0], true);
  assert.equal(sent[1].op, "view");
  assert.deepEqual(JSON.parse(sent[1].view), { layer: 1 });
});

test("a note from an old version gives a conflict", async () => {
  const state = { error: "conflict", seq: 5, expires: 0, members: [], view: "", view_by: "",
    notes: [{ neuron: "1.2", text: "theirs", by: "Ben", version: 5 }] };
  const { sent, fetch, release } = relay([{ status: 200, data: { seq: 4, expires: 0, members: [] } }, { status: 409, data: state }]);
  const conflicts = [];
  const r = joinRoom({ url: "room", room: ROOM, member: "member01", name: () => "Ada", secret: () => "",
    onState() {}, onConflict: (...a) => conflicts.push(a), onError() {}, fetch, interval: 10 });
  await until(() => sent.length);
  r.note("1.2", "mine", 3);
  release();
  await until(() => conflicts.length);
  r.leave(() => {});
  assert.equal(sent[1].op, "note");
  assert.equal(conflicts[0][0], "1.2");
  assert.equal(conflicts[0][1], "mine");
  assert.equal(conflicts[0][2].notes[0].text, "theirs");
});

test("a room that ended stops the polls", async () => {
  const { sent, fetch, release } = relay([{ status: 404, data: { error: "gone", message: "This room has ended." } }]);
  release();
  const errors = [];
  joinRoom({ url: "room", room: ROOM, member: "member01", name: () => "Ada", secret: () => "",
    onState() {}, onConflict() {}, onError: (e) => errors.push(e.error), fetch, interval: 5 });
  await until(() => errors.length);
  await new Promise((r) => setTimeout(r, 50));
  assert.deepEqual(errors, ["gone"]);
  assert.equal(sent.length, 1);
});
