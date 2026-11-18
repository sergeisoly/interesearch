/* A 3D schematic for one matrix. All readouts come from the vectors drawn here. */
(() => {
  "use strict";
  const byId = id => document.getElementById(id);
  const live = byId("repair-method-live");
  if (!live || !window.ResearchOrbit) return;

  const input = [0.8, -0.5, 0.8];
  const donorRead = [1, 0, 0];
  const donorWrite = [0.8, 0, 0.6];
  // A fixed example of a nonzero initial input vector. Unit read vectors make
  // aᵀx equal to the signed projection length in this schematic.
  const freshRead = [-0.36, 0.8, 0.48].map(v => v / Math.hypot(-0.36, 0.8, 0.48));
  const origin = [0, 0, 0];
  const colors = { input: "#285a9b", direction: "#9c4a23", output: "#187163", guide: "#627078" };
  const inputSvg = byId("repair-read-scene");
  const outputSvg = byId("repair-write-scene");
  const sliders = Object.fromEntries(["coefficient", "read-angle", "write-angle", "length"].map(key => [key, byId("repair-" + key)]));
  const methodCopy = {
    strengths: "Both donor vectors stay fixed.",
    donor_init: "Both vectors start at their donor values.",
    fresh: "The output starts at zero. Increase its length to set its direction."
  };
  let method = "strengths";
  const scale = (v, factor) => v.map(x => x * factor);
  const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
  const degrees = value => Number(value) * Math.PI / 180;
  const number = value => (Math.abs(value) < .005 ? "0" : value.toFixed(2)).replace("-", "−");
  const tuple = vector => "(" + vector.map(number).join(", ") + ")";

  function geometry() {
    const scalarOnly = method === "strengths";
    const theta = scalarOnly ? 0 : degrees(sliders["read-angle"].value);
    const phi = scalarOnly ? 0 : degrees(sliders["write-angle"].value);
    const [ax, ay, az] = method === "fresh" ? freshRead : donorRead;
    const a = [ax * Math.cos(theta) - ay * Math.sin(theta), ax * Math.sin(theta) + ay * Math.cos(theta), az];
    const [bx, by, bz] = method === "fresh" ? [0, 0.8, 0.6] : donorWrite;
    // The write direction rotates about a second axis. When its length is zero,
    // a fresh repair has no direction, and the output panel draws no line.
    const direction = [bx * Math.cos(phi) + bz * Math.sin(phi), by, bz * Math.cos(phi) - bx * Math.sin(phi)];
    const b = scale(direction, scalarOnly ? 1 : Number(sliders.length.value));
    const readout = dot(a, input);
    const coefficient = scalarOnly ? Number(sliders.coefficient.value) : 1;
    return { a, b, direction, readout, correction: scale(b, readout * coefficient) };
  }

  function draw(camera) {
    const { a, b, direction, readout, correction } = geometry();
    // Keep both panels on one scale. Allow space for endpoint labels when an
    // unusually large correction or a steep camera angle approaches the edge.
    const extent = [input, a, b, correction, scale(a, 1.35), scale(a, -1.35),
      scale(direction, 1.6), scale(direction, -1.6), [0, 0, 1.15],
      [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]];
    const views = extent.map(v => ResearchOrbit.project(v, camera));
    const size = Math.min(100, 136 / Math.max(...views.map(([x]) => Math.abs(x))),
      91 / Math.max(...views.map(([, y]) => Math.abs(y))));
    const point = vector => {
      const [x, y] = ResearchOrbit.project(vector, camera);
      return [160 + size * x, 108 + size * y];
    };
    const projected = vector => point(vector).join(",");
    function element(svg, tag, attrs, text) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
      if (text !== undefined) el.textContent = text;
      svg.append(el);
      return el;
    }
    function line(svg, start, end, color, width, dash) {
      const attrs = { points: projected(start) + " " + projected(end), fill: "none", stroke: color, "stroke-width": width };
      if (dash) attrs["stroke-dasharray"] = dash;
      element(svg, "polyline", attrs);
    }
    function circle(svg, v, r, color, fill) {
      const [cx, cy] = point(v);
      element(svg, "circle", { cx, cy, r, stroke: color, fill, "stroke-width": 1.8 });
    }
    function arrow(svg, v, color) {
      const [sx, sy] = point(origin);
      const [x, y] = point(v);
      const length = Math.hypot(x - sx, y - sy);
      if (length < 2.5) { circle(svg, origin, 3.6, color, color); return; }
      const dx = (x - sx) / length, dy = (y - sy) / length;
      const head = Math.min(10, length * .6);
      const hx = x - head * dx, hy = y - head * dy;
      element(svg, "line", { x1: sx, y1: sy, x2: hx, y2: hy, stroke: color, "stroke-width": 3.3 });
      element(svg, "polygon", { points: [[x,y], [hx - .42 * head * dy, hy + .42 * head * dx], [hx + .42 * head * dy, hy - .42 * head * dx]].map(p => p.join(",")).join(" "), fill: color });
    }
    function label(svg, v, word, dx, dy, color) {
      const [x, y] = point(v);
      element(svg, "text", { x: x + dx, y: y + dy, fill: color, "font-size": 14,
        "font-weight": 600, "font-style": "italic", "text-anchor": dx < 0 ? "end" : "start",
        stroke: "var(--research-paper, #faf9f5)", "stroke-width": 3.5, "paint-order": "stroke" }, word);
    }
    function frame(svg) {
      svg.replaceChildren();
      for (const offset of [-1, -.5, 0, .5, 1]) {
        const color = offset === 0 ? "#a4ada9" : "#d7ddda";
        line(svg, [offset, -1, 0], [offset, 1, 0], color, 1);
        line(svg, [-1, offset, 0], [1, offset, 0], color, 1);
      }
      line(svg, origin, [0, 0, 1.15], "#a4ada9", 1);
      circle(svg, origin, 2.6, colors.guide, colors.guide);
    }
    frame(inputSvg);
    line(inputSvg, scale(a, -1.35), scale(a, 1.35), colors.direction, 1.6, "4 4");
    line(inputSvg, input, scale(a, readout), colors.guide, 1.7, "3 3");
    arrow(inputSvg, a, colors.direction);
    arrow(inputSvg, input, colors.input);
    const [px, py] = point(scale(a, readout));
    element(inputSvg, "rect", { x: px - 3.5, y: py - 3.5, width: 7, height: 7, stroke: colors.guide, fill: "#fff", "stroke-width": 1.5 });
    label(inputSvg, input, "x", 6, -6, colors.input);
    label(inputSvg, a, "a", 6, 15, colors.direction);
    frame(outputSvg);
    if (Math.hypot(...b) > 0) {
      line(outputSvg, scale(direction, -1.6), scale(direction, 1.6), colors.direction, 1.8, "5 4");
      circle(outputSvg, b, 5, colors.direction, "#fff");
    }
    arrow(outputSvg, correction, colors.output);
    if (Math.hypot(...b) > 0) label(outputSvg, b, "b", 7, 13, colors.direction);
    label(outputSvg, correction, "Ex", -7, -6, colors.output);
    byId("repair-scalar-value").textContent = number(readout);
    byId("repair-vector-value").textContent = tuple(correction);
    inputSvg.setAttribute("aria-label", "Input space. Input x " + tuple(input) + ", read vector a " + tuple(a) + ", signed projection " + number(readout) + ". Arrow keys rotate both views.");
    outputSvg.setAttribute("aria-label", "Output space. Write vector b " + tuple(b) + ", added output " + tuple(correction) + ". Arrow keys rotate both views.");
  }

  const orbit = ResearchOrbit.bind([inputSvg, outputSvg], draw, { yaw: .6, pitch: .35 });
  function update() {
    for (const [key, slider] of Object.entries(sliders)) {
      const isAngle = key.endsWith("angle");
      const description = isAngle ? slider.value + "°" : number(Number(slider.value));
      slider.parentElement.querySelector("output").value = description;
      slider.setAttribute("aria-valuetext", description);
    }
    sliders["write-angle"].disabled = method !== "strengths" && Number(sliders.length.value) === 0;
    orbit.redraw();
  }
  function select(next) {
    method = next;
    const scalarOnly = method === "strengths";
    for (const [key, slider] of Object.entries(sliders)) {
      slider.parentElement.hidden = scalarOnly !== (key === "coefficient");
      slider.value = key === "coefficient" || (key === "length" && method !== "fresh") ? "1" : "0";
    }
    for (const button of live.querySelectorAll("[data-repair-method]")) {
      button.setAttribute("aria-pressed", String(button.dataset.repairMethod === method));
    }
    const hint = byId("repair-method-copy");
    hint.textContent = methodCopy[method];
    hint.classList.toggle("visually-hidden", method !== "fresh");
    update();
  }
  live.querySelectorAll("[data-repair-method]").forEach(button => button.addEventListener("click", () => select(button.dataset.repairMethod)));
  Object.values(sliders).forEach(slider => slider.addEventListener("input", update));
  byId("repair-view-reset").addEventListener("click", () => orbit.reset());
  select(method);
  live.hidden = false;
  byId("repair-method-fallback").hidden = true;
})();
