// Apply the same toy matrices as the cell view to three example inputs.
window.initMatrixOutput = function (coefficient, toy) {
  const panel = document.getElementById("matrix-output");
  const svg = document.getElementById("matrix-output-space");
  const reading = document.getElementById("matrix-output-reading");
  const select = document.getElementById("matrix-input");
  const switcher = document.getElementById("matrix-view-switch");
  const inputs = [[1, 0, 0], [0, 1, 0], [1, 2, 0]];
  const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
  const fmt = x => String(Number(x.toFixed(1))).replace("-", "−");

  function draw(camera) {
    if (panel.hidden) return;
    const x = inputs[select.selectedIndex];
    const read = dot(toy.v, x);
    const rest = toy.remainder.map(row => dot(row, x));
    const at = c => rest.map((v, i) => v + c * toy.u[i] * read);
    const current = at(Number(coefficient.value));
    const width = svg.clientWidth || 400;
    const height = Math.max(210, Math.min(305, width * 0.56));
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const axes = [[4, 0, 0], [0, 4, 0], [0, 0, 4]];
    const project = v => window.ResearchOrbit.project(v, camera);
    const bounds = [at(-1), at(1), ...axes, [-2,-2,0], [-2,4,0], [4,-2,0], [4,4,0]].map(project);
    const xs = bounds.map(p => p[0]), ys = bounds.map(p => p[1]);
    const left = Math.min(...xs), right = Math.max(...xs);
    const top = Math.min(...ys), bottom = Math.max(...ys);
    const scale = Math.min((width - 54) / (right - left), (height - 44) / (bottom - top));
    function screen(v) {
      const [x, y] = project(v);
      return [width / 2 + scale * (x - (left + right) / 2), height / 2 + scale * (y - (top + bottom) / 2)];
    }
    const fragment = document.createDocumentFragment();
    function add(name, attrs, label) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key,value] of Object.entries(attrs)) el.setAttribute(key,value);
      if (label) el.textContent = label;
      fragment.append(el);
    }
    function line(a, b, color, thickness, dash) {
      const [x1,y1] = screen(a), [x2,y2] = screen(b);
      add("line", { x1, y1, x2, y2, stroke: color, "stroke-width": thickness, "stroke-dasharray": dash || "none" });
    }
    for (let k = -2; k <= 4; k += 2) {
      line([k,-2,0], [k,4,0], "#dedbd3", 1);
      line([-2,k,0], [4,k,0], "#dedbd3", 1);
    }
    const origin = [0,0,0];
    axes.forEach((axis, i) => {
      line(origin, axis, "#8e928c", 1.3);
      const [x,y] = screen(axis);
      add("text", {x: x + 5, y: y - 5, fill: "#545e60", "font-size": 14}, ["y₁","y₂","y₃"][i]);
    });
    line(origin, rest, "#9a9c99", 1.3);
    line(at(-1), at(1), "#9e7559", 2, "5 4");
    line([current[0], current[1], 0], current, "#9a9c99", 1, "2 3");
    line(origin, current, "#246c81", 2.5);
    const [zx,zy] = screen(rest), [ox,oy] = screen(at(1)), [cx,cy] = screen(current);
    add("rect", {x:zx-7, y:zy-7, width:14, height:14, fill:"none", stroke:"#616567", "stroke-width":2});
    add("path", {d:`M ${ox} ${oy-11} l 11 11 -11 11 -11 -11 z`, fill:"none", stroke:"#985a29", "stroke-width":2});
    add("circle", {cx, cy, r:4.8, fill:"#246c81", stroke:"#fff", "stroke-width":1.4});
    svg.replaceChildren(fragment);
    const explanation = read === 0
      ? "This input is perpendicular to the read vector, so every coefficient gives the same output."
      : "Changing the coefficient moves the output along the dashed line.";
    const text = `Input read vᵀx = ${fmt(read)}. Output = (${current.map(fmt).join(", ")}). ${explanation}`;
    if (reading.textContent !== text) reading.textContent = text;
  }

  const orbit = window.ResearchOrbit.bind([svg], draw, {yaw: 0.64, pitch: 0.4});
  const entries = document.querySelector(".visual-matrix-equation");
  const matrixState = document.getElementById("visual-matrix-state");
  switcher.addEventListener("change", () => {
    const showOutput = switcher.querySelector(":checked").value === "output";
    panel.hidden = !showOutput;
    entries.hidden = showOutput;
    matrixState.hidden = showOutput;
    document.getElementById("visual-matrix-key").hidden = showOutput;
    coefficient.setAttribute("aria-describedby", showOutput ? reading.id : matrixState.id);
    orbit.redraw();
  });
  document.getElementById("matrix-reset").addEventListener("click", orbit.reset);
  select.addEventListener("change", orbit.redraw);
  coefficient.addEventListener("input", orbit.redraw);
  if (window.ResizeObserver) new ResizeObserver(orbit.redraw).observe(panel);
  else window.addEventListener("resize", orbit.redraw);
  switcher.hidden = false;
};
