// Screenshots of the neurons page: before and after, at 390 and 1280 px, rest, hover, keyboard focus and press.
import { readFileSync, writeFileSync } from "node:fs";
const [port, out, css] = [process.argv[2], process.argv[3], readFileSync(process.argv[4], "utf8")];
const t = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
const ws = new WebSocket(t.webSocketDebuggerUrl); await new Promise((r) => (ws.onopen = r));
let id = 0; const wait = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (wait.has(d.id)) { wait.get(d.id)(d); wait.delete(d.id); } };
const cdp = (method, params = {}) => new Promise((r) => { const i = ++id; wait.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const js = async (e) => (await cdp("Runtime.evaluate", { expression: e, returnByValue: true, awaitPromise: true })).result?.result?.value;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const shot = async (name) => { const d = await cdp("Page.captureScreenshot", { format: "png" }); writeFileSync(`${out}/${name}.png`, Buffer.from(d.result.data, "base64")); };
await cdp("Page.enable");
for (const w of [1280, 390]) {
  await cdp("Emulation.setDeviceMetricsOverride", { width: w, height: 900, deviceScaleFactor: 1, mobile: w < 500 });
  for (const [v, url] of [["before", "http://localhost:18732/"], ["after", "http://localhost:18731/"]]) {
    await cdp("Page.navigate", { url }); await sleep(2500);
    if (v === "after") await js(`(() => { const s = document.createElement('style'); s.textContent = ${JSON.stringify(css)}; document.head.append(s); })()`);
    await js(`document.querySelector('.together').scrollIntoView()`); await sleep(300);
    await shot(`${v}-${w}-rest`);
    // keyboard focus: Tab from the name field to the "Share a link" button
    await js(`document.getElementById('roomName').focus()`);
    await cdp("Input.dispatchKeyEvent", { type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
    await cdp("Input.dispatchKeyEvent", { type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
    await sleep(400); console.log(v, w, await js(`document.activeElement.id + ' ' + getComputedStyle(document.activeElement).boxShadow.slice(0, 40)`));
    await shot(`${v}-${w}-focus`);
    await js(`document.activeElement.blur()`);
    const r = await js(`(() => { const b = document.getElementById('share').getBoundingClientRect(); return [b.x + b.width / 2, b.y + b.height / 2]; })()`);
    await cdp("Input.dispatchMouseEvent", { type: "mouseMoved", x: r[0], y: r[1] }); await sleep(400);
    await shot(`${v}-${w}-hover`);
    await cdp("Input.dispatchMouseEvent", { type: "mousePressed", x: r[0], y: r[1], button: "left", clickCount: 1 }); await sleep(300);
    console.log(v, w, "press", await js(`getComputedStyle(document.getElementById('share')).transform`));
    await shot(`${v}-${w}-press`);
    await js(`document.querySelector('.share') ; document.getElementById('leave')`);
    await cdp("Input.dispatchMouseEvent", { type: "mouseMoved", x: 1, y: 1 });
    // release away from the button so it does not make a room
    await cdp("Input.dispatchMouseEvent", { type: "mouseReleased", x: 1, y: 1, button: "left", clickCount: 1 });
    await js(`document.querySelector('.run').scrollIntoView()`); await sleep(300);
    await shot(`${v}-${w}-run`);
    await js(`document.querySelector('.ask').scrollIntoView()`); await sleep(300);
    await shot(`${v}-${w}-ask`);
  }
}
for (const [n, url] of [["ref-sonner", "https://sonner.emilkowal.ski/"], ["ref-animations", "https://animations.dev/"]]) {
  await cdp("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  await cdp("Page.navigate", { url }); await sleep(4000); await shot(`${n}-1280`);
}
await cdp("Page.close"); process.exit(0);
