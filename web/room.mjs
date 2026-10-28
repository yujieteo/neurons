// The rooms: a share link opens a room on the relay of teoyujie.org, and
// 2 to 5 people see the same view and the same notes (README.md, Rooms).
// The page polls the relay once each second, with one request at a time.

// The id of a room (16 bytes, 22 characters) or of a member (9 bytes, 12
// characters), in base64url.
export function newId(bytes = 16) {
  const b = crypto.getRandomValues(new Uint8Array(bytes));
  return btoa(String.fromCharCode(...b)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export const roomOfHash = (hash) => /^#room=([A-Za-z0-9_-]{22})$/.exec(hash)?.[1] ?? null;

// The fields that each operation sends, after op, room, member, name and since.
const FIELDS = { create: [], poll: [], leave: [], view: ["view"], note: ["neuron", "text", "base"] };

export class KeyError extends Error {
  constructor() { super("This looks like an OpenRouter key. The page does not send keys to the room."); }
}

// The body of one request. Only the fields of the protocol go in. A body that
// holds the visitor's key, or text like an OpenRouter key, is refused.
export function body(op, room, member, name, since, fields, secret) {
  const msg = { op, room, member, name, since };
  for (const f of FIELDS[op]) msg[f] = fields[f];
  const text = JSON.stringify(msg);
  if (/sk-or-/i.test(text) || (secret && secret.length >= 8 && text.includes(secret))) throw new KeyError();
  return text;
}

// Joins room, or makes it first when create is true. The callbacks:
// onState(state) after each reply, onConflict(neuron, text, state) when a note
// changed before the save of this member came, and onError({ error, message }).
// name() and secret() give the name of the member and the visitor's key now.
export function joinRoom({ url, room, member, name, secret, create = false, onState, onConflict, onError,
  fetch = globalThis.fetch, interval = 1000 }) {
  let since = -1, stopped = false, timer = null, inFlight = false;
  let view = null;               // the latest view that is not sent yet
  const notes = [];              // the notes that are not sent yet, in sequence

  const schedule = (ms) => {
    clearTimeout(timer);
    if (!stopped) timer = setTimeout(step, ms);
  };

  async function step() {
    timer = null;
    if (stopped || inFlight) return;
    let op = create ? "create" : "poll", fields = {};
    if (!create && view !== null) { op = "view"; fields = { view }; view = null; }
    else if (!create && notes.length) { op = "note"; fields = notes.shift(); }
    let text;
    try { text = body(op, room, member, name(), since, fields, secret()); } catch (e) {
      onError({ error: "key", message: e.message, op });
      schedule(op === "poll" || op === "create" ? interval : 0);
      return;
    }
    inFlight = true;
    let res, data;
    try {
      res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: text });
      data = await res.json();
    } catch {
      inFlight = false;
      if (op === "view" && view === null) view = fields.view;
      if (op === "note") notes.unshift(fields);
      onError({ error: "net", message: "The page cannot reach the room. It tries again." });
      schedule(5 * interval);
      return;
    }
    inFlight = false;
    if (stopped) return;
    if (res.ok || data.error === "conflict") {
      if (op === "create") create = false;
      if (data.notes) since = data.seq;
      onState(data);
      if (data.error === "conflict") onConflict(fields.neuron, fields.text, data);
      schedule(view !== null || notes.length ? 0 : interval);
    } else {
      onError({ ...data, op });
      if (data.error === "gone" || data.error === "full" || data.error === "exists") stopped = true;
      else schedule(res.status === 429 ? 10 * interval : interval);
    }
  }

  const kick = () => { if (!inFlight) schedule(0); };
  schedule(0);
  return {
    // A view that waits to go out is newer than the view in a reply.
    viewPending: () => view !== null,
    setView(v) { view = JSON.stringify(v); kick(); },
    note(neuron, text, base) { notes.push({ neuron, text, base }); kick(); },
    leave(beacon = globalThis.navigator?.sendBeacon?.bind(globalThis.navigator)) {
      stopped = true;
      clearTimeout(timer);
      try { beacon?.(url, body("leave", room, member, name(), since, {}, secret())); } catch { /* a key in the name */ }
    },
  };
}
