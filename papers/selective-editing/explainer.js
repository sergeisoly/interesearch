/* Presentation only. Measurements and intervals come from the paper's saved results. */
(() => {
  "use strict";
  const source = document.getElementById("visual-selective-data");
  if (!source) return;
  let data;
  try { data = JSON.parse(source.textContent); } catch (_) { return; }
  const NS = "http://www.w3.org/2000/svg";
  const methods = [
    {id: "vpd_atom", label: "One VPD subcomponent", color: "#2a78d6", shape: "circle"},
    {id: "lora_rank1", label: "Rank-one LoRA", color: "#c85224", shape: "square"},
    {id: "finetune_24", label: "Full-matrix fine-tuning", color: "#4a3aa7", shape: "triangle"},
    {id: "neurons", label: "Neuron edit", color: "#12865f", shape: "diamond"},
  ];
  const pairs = {de: ["German", "Dutch"], es: ["Spanish", "Portuguese"], fr: ["French", "Italian"]};
  const state = {target: "de", tau: 1};
  const el = (tag, attrs = {}, text) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  // Roughly two significant figures. Small nonzero effects never become zero.
  const number = value => {
    if (value === 0) return "0";
    const a = Math.abs(value);
    const digits = a >= 0.1 ? 2 : a >= 0.01 ? 3 : a >= 0.001 ? 4 : Math.max(0, 1 - Math.floor(Math.log10(a)));
    return value.toFixed(digits).replace("-", "−");
  };
  const signed = value => `${value > 0 ? "+" : ""}${number(value)}`;
  const range = values => {
    const lo = number(Math.min(...values)), hi = number(Math.max(...values));
    return lo === hi ? lo : `${lo}–${hi}`;
  };
  const mark = (svg, method, x, y, title) => {
    let point;
    const attrs = {fill: method.color, stroke: "white", "stroke-width": 1};
    if (method.shape === "square") point = el("rect", {...attrs, x: x - 4.5, y: y - 4.5, width: 9, height: 9});
    else if (method.shape === "triangle") point = el("path", {...attrs, d: `M ${x} ${y-6} l 5.5 10 h -11 z`});
    else if (method.shape === "diamond") point = el("path", {...attrs, d: `M ${x} ${y-6} l 6 6 -6 6 -6 -6 z`});
    else point = el("circle", {...attrs, cx: x, cy: y, r: 5});
    point.append(el("title", {}, title));
    svg.append(point);
  };
  function plot(field, language, targetPanel, edits) {
    const container = document.getElementById(targetPanel ? "visual-target-chart" : "visual-protected-chart");
    const width = Math.max(280, container.clientWidth || 470);
    const compact = width < 430;
    const height = compact ? 386 : 324;
    const svg = el("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${language}: ${targetPanel ? "target effect" : "related-language damage"}, cross-entropy rise in nat per token`});
    svg.append(el("text", {x: 16, y: 28, fill: "#243c3d", "font-size": 20, "font-weight": 600}, language));
    svg.append(el("text", {x: 16, y: 50, fill: "#53625d", "font-size": 14}, targetPanel ? "Target language · measured damage" : "Related language · less is better"));
    const left = compact ? 16 : 188, right = width - 19;
    const x = value => left + value / 1.2 * (right - left);
    const top = compact ? 108 : 76, bottom = compact ? 322 : 270;
    [0, 0.5, 1].forEach(tick => {
      svg.append(el("line", {x1: x(tick), x2: x(tick), y1: top, y2: bottom, stroke: "#dce3df", "stroke-width": 1}));
      svg.append(el("text", {x: x(tick), y: compact ? 351 : 293, "text-anchor": "middle", fill: "#53625d", "font-size": 13}, tick.toString()));
    });
    if (targetPanel) {
      svg.append(el("line", {x1: x(state.tau), x2: x(state.tau), y1: top, y2: bottom, stroke: "#70817b", "stroke-dasharray": "4 4", "stroke-width": 1.3}));
    }
    methods.forEach((method, index) => {
      const rows = edits.filter(e => e.method === method.id);
      const values = rows.map(e => e[field]);
      const y = compact ? 120 + index * 65 : 102 + index * 49;
      if (compact) svg.append(el("rect", {x: 0, y: y - 38, width, height: 25, fill: "#faf9f5"}));
      svg.append(el("text", {x: 16, y: compact ? y - 21 : y - 4, fill: "#344946", "font-size": 14}, method.label));
      svg.append(el("text", {x: compact ? right : 16, y: compact ? y - 21 : y + 15, "text-anchor": compact ? "end" : "start", fill: "#344946", "font-size": 14, "font-weight": 600}, range(values)));
      svg.append(el("line", {x1: x(Math.min(...values)), x2: x(Math.max(...values)), y1: y, y2: y, stroke: method.color, "stroke-width": 2}));
      const visibleRows = method.id === "vpd_atom" ? rows.slice(0, 1) : rows;
      visibleRows.forEach(row => mark(svg, method, x(row[field]), y, `${method.label}${method.id === "vpd_atom" ? "" : `, training subset ${row.draw + 1}`}: ${number(row[field])} nat/token`));
    });
    svg.append(el("text", {x: right, y: height - 5, "text-anchor": "end", fill: "#53625d", "font-size": 13}, "Prediction loss increase · nat/token"));
    const accessibleValues = methods.map(method => `${method.label}: ${range(edits.filter(e => e.method === method.id).map(e => e[field]))} nat per token`).join(". ");
    svg.setAttribute("aria-label", `${language}, ${targetPanel ? "target effect" : "related-language damage"}. Cross-entropy rise relative to the unedited model. ${accessibleValues}. Both panels use the same zero to 1.2 nat per token scale.`);
    return svg;
  }
  function englishPlot(edits) {
    const container = document.getElementById("visual-english-chart");
    const width = Math.max(280, container.clientWidth || 470);
    const compact = width < 430;
    const height = compact ? 386 : 324;
    const svg = el("svg", {viewBox: `0 0 ${width} ${height}`, role: "img"});
    svg.append(el("text", {x: 16, y: 28, fill: "#243c3d", "font-size": 20, "font-weight": 600}, "Change in English predictions"));
    svg.append(el("text", {x: 16, y: 51, fill: "#53625d", "font-size": 14}, `${pairs[state.target][0]} edit · ${state.tau} nat/token target · lower is better`));
    const left = compact ? 30 : 188, right = width - 30;
    const x = value => left + (Math.log10(value) + 5) / 4 * (right - left);
    const top = compact ? 108 : 78, bottom = compact ? 321 : 273;
    [0.00001, 0.0001, 0.001, 0.01, 0.1].forEach(tick => {
      svg.append(el("line", {x1: x(tick), x2: x(tick), y1: top, y2: bottom, stroke: "#dce3df", "stroke-width": 1}));
      svg.append(el("text", {x: x(tick), y: compact ? 347 : 296, "text-anchor": "middle", fill: "#53625d", "font-size": compact ? 11 : 13}, tick.toString()));
    });
    methods.forEach((method, index) => {
      const rows = edits.filter(row => row.method === method.id);
      const values = rows.map(row => row.english_kl);
      if (values.some(value => !Number.isFinite(value) || value < 0.00001 || value > 0.1)) throw new Error("English KL outside plotted scale");
      const y = compact ? 120 + index * 65 : 104 + index * 49;
      if (compact) svg.append(el("rect", {x: 0, y: y - 38, width, height: 25, fill: "#faf9f5"}));
      svg.append(el("text", {x: 16, y: compact ? y - 21 : y - 4, fill: "#344946", "font-size": 14}, method.label));
      svg.append(el("text", {x: compact ? width - 16 : 16, y: compact ? y - 21 : y + 15, "text-anchor": compact ? "end" : "start", fill: "#344946", "font-size": 14, "font-weight": 600}, range(values)));
      svg.append(el("line", {x1: x(Math.min(...values)), x2: x(Math.max(...values)), y1: y, y2: y, stroke: method.color, "stroke-width": 2}));
      const visibleRows = method.id === "vpd_atom" ? rows.slice(0, 1) : rows;
      visibleRows.forEach(row => mark(svg, method, x(row.english_kl), y, `${method.label}${method.id === "vpd_atom" ? "" : `, training subset ${row.draw + 1}`}: English KL ${number(row.english_kl)} nat/token`));
    });
    svg.append(el("text", {x: right, y: height - 5, "text-anchor": "end", fill: "#53625d", "font-size": 13}, "KL divergence · nat/token · log scale"));
    const values = methods.map(method => `${method.label}: ${range(edits.filter(row => row.method === method.id).map(row => row.english_kl))} nat per token`).join(". ");
    svg.setAttribute("aria-label", `English predictions after the ${pairs[state.target][0]} edit at a ${state.tau} nat per token calibration target. KL divergence from the original distribution to the edited one, averaged over English text on ${data.n_dates} evaluation dates. Lower is closer to the original model. ${values}. Logarithmic scale from 0.00001 to 0.1, with the same scale for every language and strength.`);
    return svg;
  }
  function uncertainty(rows) {
    const table = document.createElement("table");
    table.className = "visual-uncertainty-table";
    const caption = table.createCaption();
    caption.textContent = `${pairs[state.target].join(" → ")}, target loss increase on validation text: ${state.tau} nat/token. Subcomponent edit minus comparator: estimate [simultaneous 95% interval], nat/token.`;
    const head = table.createTHead().insertRow();
    ["Comparator", "Related-language difference", "Achieved target gap (post hoc)"].forEach(text => {
      const th = document.createElement("th"); th.scope = "col"; th.textContent = text; head.append(th);
    });
    const body = table.createTBody();
    methods.slice(1).forEach(method => {
      const row = rows.find(r => r.comparator === method.id);
      const tr = body.insertRow(), th = document.createElement("th");
      th.scope = "row"; th.textContent = method.label; tr.append(th);
      [row.protected, row.target_gap].forEach((interval, index) => {
        // The ±0.1 agreement boundary needs three decimals: 0.104 must not read as 0.10.
        const format = index === 1 ? value => `${value > 0 ? "+" : ""}${Math.abs(value) < 0.001 ? number(value) : value.toFixed(3).replace("-", "−")}` : signed;
        tr.insertCell().textContent = `${format(interval.estimate)} [${format(interval.lo)}, ${format(interval.hi)}]`;
      });
    });
    return table;
  }
  function render() {
    const edits = data.edits.filter(e => e.target === state.target && e.tau === state.tau);
    const comparisons = data.comparisons.filter(e => e.target === state.target && e.tau === state.tau);
    if (edits.length !== 12 || comparisons.length !== 3) throw new Error("Incomplete presentation data");
    const [target, protectedLanguage] = pairs[state.target];
    document.getElementById("visual-target-chart").replaceChildren(plot("target_rise", target, true, edits));
    document.getElementById("visual-protected-chart").replaceChildren(plot("protected_rise", protectedLanguage, false, edits));
    document.getElementById("visual-english-chart").replaceChildren(englishPlot(edits));
    const atom = edits.find(e => e.method === "vpd_atom");
    document.getElementById("visual-result-sentence").textContent = `Changing one VPD subcomponent's coefficient worsened ${target} next-token prediction while leaving average prediction loss on ${protectedLanguage} text almost unchanged. On the evaluation text, loss rose by ${number(atom.target_rise)} nat/token for ${target} but only ${number(atom.protected_rise)} nat/token for ${protectedLanguage}.`;
    document.getElementById("visual-uncertainty").replaceChildren(uncertainty(comparisons));
    document.querySelectorAll("[data-visual-tau]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.visualTau) === state.tau)));
    ["visual-pair", "visual-english-pair"].forEach(id => { document.getElementById(id).value = state.target; });
  }
  // The matrix toy is deliberately independent of every measured result above.
  function matrixIllustration(coefficient) {
    const remainder = [[0.3, -0.2, 0.5], [0.1, 0.6, -0.3], [-0.3, 0.1, 0.4]];
    const u = [0.7, -0.5, 0.3], v = [0.8, -0.6, 0.4];
    const atom = u.map(a => v.map(b => a * b));
    const values = [remainder, atom.map(row => row.map(value => coefficient * value)), remainder.map((row, i) => row.map((value, j) => value + coefficient * atom[i][j]))];
    values.forEach((matrix, index) => {
      const group = document.getElementById(`visual-matrix-cells-${index}`);
      group.replaceChildren();
      const left = 4;
      matrix.forEach((row, i) => row.forEach((value, j) => {
        const rect = el("rect", {x: left + j * 43, y: 4 + i * 43, width: 40, height: 40, rx: 2, fill: value < 0 ? "#826aab" : "#327e70", "fill-opacity": Math.max(0.035, Math.abs(value))});
        group.append(rect);
      }));
      group.append(el("rect", {x: left - 2, y: 2, width: 130, height: 130, rx: 3, fill: "none", stroke: "#bac6bf"}));
    });
    const phrase = coefficient === 1 ? "The original coefficient: no edit." : coefficient === 0 ? "The selected subcomponent is removed." : coefficient < 0 ? "The selected subcomponent contributes with its sign reversed." : "The selected subcomponent contributes less than before.";
    document.getElementById("visual-matrix-state").textContent = phrase;
    document.getElementById("visual-coefficient-value").value = number(coefficient);
  }
  try {
    document.getElementById("visual-results-interactive").hidden = false;
    document.getElementById("visual-english-interactive").hidden = false;
    render();
    matrixIllustration(0);
    document.getElementById("visual-results-fallback").hidden = true;
    document.getElementById("visual-english-fallback").hidden = true;
    document.getElementById("visual-coefficient-control").hidden = false;
    ["visual-pair", "visual-english-pair"].forEach(id => document.getElementById(id).addEventListener("change", event => { state.target = event.target.value; render(); }));
    document.querySelectorAll("[data-visual-tau]").forEach(button => button.addEventListener("click", () => { state.tau = Number(button.dataset.visualTau); render(); }));
    document.getElementById("visual-coefficient").addEventListener("input", event => matrixIllustration(Number(event.target.value)));
    let frame;
    const redraw = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(render);
    };
    const charts = document.querySelector(".visual-chart-pair");
    if (typeof ResizeObserver !== "undefined") {
      let lastWidth = charts.clientWidth;
      new ResizeObserver(() => {
        if (charts.clientWidth !== lastWidth) {
          lastWidth = charts.clientWidth;
          redraw();
        }
      }).observe(charts);
    } else {
      window.addEventListener("resize", redraw);
    }
  } catch (error) {
    document.getElementById("visual-results-interactive").hidden = true;
    document.getElementById("visual-results-fallback").hidden = false;
    document.getElementById("visual-english-interactive").hidden = true;
    document.getElementById("visual-english-fallback").hidden = false;
    // Keep the complete static example available when a build has missing data.
    console.warn("Selective-editing visual could not initialize.", error);
  }
})();
