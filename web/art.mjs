// The layers, drawn in the manner of the art on teoyujie.org: one rule per
// layer from left to right, the neurons as points on it, a path through the
// strongest neuron of each layer that runs in from the left, and at the
// right the histogram of all the neuron values beside the normal density.
//
// A point's height is its value as a z-score, (value - mean) / std over the
// sweep, on a log scale up to ZMAX.

const ZMAX = 1000;
const RUN = 1.2;      // seconds for the path to cross the layers
const BINS = 26;      // histogram bins from Z0 to Z1
const Z0 = -3, Z1 = 10;

export function createLayers(canvas, layers, onPick) {
  const ctx = canvas.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  const still = matchMedia("(prefers-reduced-motion: reduce)");
  let state = { points: [], path: [], selected: null, hist: null };
  let start = 0, frame = 0, hits = [];

  function geometry() {
    const w = canvas.width, h = canvas.height;
    const x0 = 0.04 * w, x1 = 0.78 * w, top = 0.08 * h, base = 0.86 * h;
    return {
      w, h, unit: w / 400, x0, x1, top, base,
      x: (l) => x0 + (x1 - x0) * l / (layers - 1),
      y: (z) => base - (base - top) * Math.log1p(Math.min(Math.max(z, 0), ZMAX)) / Math.log1p(ZMAX),
    };
  }

  function draw(now) {
    const g = geometry(), { unit } = g;
    const ink = color("--ink"), accent = color("--accent"), line = color("--line"), muted = color("--muted");
    const progress = still.matches ? 1 : Math.min(Math.max((now - start) / 1000 / RUN, 0), 1);
    ctx.clearRect(0, 0, g.w, g.h);
    ctx.lineWidth = unit;
    ctx.strokeStyle = line;
    for (let l = 0; l < layers; l++) {
      ctx.beginPath(); ctx.moveTo(g.x(l), g.top); ctx.lineTo(g.x(l), g.base); ctx.stroke();
    }
    ctx.fillStyle = muted;
    ctx.font = `${7 * unit}px ${color("--sans")}`;
    ctx.textAlign = "center";
    for (let l = 0; l < layers; l += 5) ctx.fillText(String(l), g.x(l), g.base + 11 * unit);
    ctx.textAlign = "left";
    ctx.fillText("layer", g.x(layers - 1) + 8 * unit, g.base + 11 * unit);

    hits = [];
    for (const p of state.points) {
      const px = g.x(p.layer), py = g.y(p.z);
      hits.push({ x: px, y: py, p });
      ctx.globalAlpha = p.mark ? 0.8 : 0.35;
      ctx.fillStyle = p.mark ? accent : ink;
      ctx.beginPath(); ctx.arc(px, py, (p.mark ? 2.6 : 2) * unit, 0, 2 * Math.PI); ctx.fill();
    }

    // The path, drawn up to its head as it runs in.
    if (state.path.length) {
      const head = progress * (state.path.length - 1), last = Math.floor(head);
      const pt = (k) => [g.x(state.path[k].layer), g.y(state.path[k].z)];
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.4 * unit;
      ctx.beginPath(); ctx.moveTo(...pt(0));
      for (let k = 1; k <= last; k++) ctx.lineTo(...pt(k));
      let hx, hy;
      if (last < state.path.length - 1) {
        const [ax, ay] = pt(last), [bx, by] = pt(last + 1), f = head - last;
        hx = ax + (bx - ax) * f; hy = ay + (by - ay) * f;
        ctx.lineTo(hx, hy);
      } else [hx, hy] = pt(last);
      ctx.stroke();
      ctx.fillStyle = accent;
      ctx.beginPath(); ctx.arc(hx, hy, 2.5 * unit, 0, 2 * Math.PI); ctx.fill();
    }

    if (state.selected) {
      const s = state.selected;
      ctx.globalAlpha = 1;
      ctx.strokeStyle = ink;
      ctx.lineWidth = 1.2 * unit;
      ctx.beginPath(); ctx.arc(g.x(s.layer), g.y(s.z), 6 * unit, 0, 2 * Math.PI); ctx.stroke();
    }

    // All the values as z-scores, against the normal density: most neurons
    // sit in the curve, and the few that fire stand out to the right.
    if (state.hist) {
      const hx = g.x1 + 0.04 * g.w, width = g.w - hx - 0.02 * g.w;
      const yc = (z) => g.base - (g.base - g.top) * (z - Z0) / (Z1 - Z0);
      const dz = (Z1 - Z0) / BINS, peak = 1 / Math.sqrt(2 * Math.PI);
      const total = state.hist.reduce((a, b) => a + b, 0);
      ctx.strokeStyle = line;
      ctx.lineWidth = unit;
      ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.moveTo(hx, g.top); ctx.lineTo(hx, g.base); ctx.stroke();
      ctx.fillStyle = accent;
      ctx.globalAlpha = 0.3;
      for (let i = 0; total && i < BINS; i++) {
        const len = Math.min(width * state.hist[i] / total / dz / peak, width);
        ctx.fillRect(hx + 2 * unit, yc(Z0 + (i + 1) * dz) + unit / 2, len, (g.base - g.top) / BINS - unit);
      }
      ctx.globalAlpha = 0.6;
      ctx.strokeStyle = ink;
      ctx.beginPath();
      for (let z = Z0; z <= Z1; z += 0.05) ctx.lineTo(hx + 2 * unit + width * Math.exp(-z * z / 2), yc(z));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (progress < 1) frame = requestAnimationFrame(draw); else frame = 0;
  }

  function redraw(animate) {
    if (animate) start = performance.now();
    if (!frame) frame = requestAnimationFrame(draw);
  }

  new ResizeObserver(() => {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    draw(performance.now() + RUN * 1000);
  }).observe(canvas);

  canvas.addEventListener("click", (ev) => {
    const r = canvas.getBoundingClientRect(), sx = canvas.width / r.width;
    const mx = (ev.clientX - r.left) * sx, my = (ev.clientY - r.top) * sx;
    let best = null, dist = 12 * sx;
    for (const h of hits) {
      const d = Math.hypot(h.x - mx, h.y - my);
      if (d < dist) { dist = d; best = h.p; }
    }
    if (best) onPick(best.layer, best.index);
  });

  return {
    // points: [{ layer, index, z, mark }], mark drawn in the accent; path: one point per layer;
    // hist: counts of all values in BINS bins, or null.
    show(points, path, hist) {
      state = { ...state, points, path, hist };
      redraw(true);
    },
    select(sel) { state.selected = sel; redraw(false); },
    bins(zs) {
      const hist = new Float64Array(BINS);
      for (const z of zs) {
        const b = Math.floor((z - Z0) / (Z1 - Z0) * BINS);
        if (b >= 0 && b < BINS) hist[b]++;
      }
      return hist;
    },
  };
}
