/* Presentation only. Measurements come from the study's saved results. */
(() => {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const methods = [
    { id: "vpd_atom", label: "One VPD subcomponent", color: "#2364a5", shape: "circle" },
    { id: "lora_rank1", label: "Rank-one LoRA", color: "#a54520", shape: "square" },
    { id: "finetune_24", label: "Full-matrix fine-tuning", color: "#5341a4", shape: "triangle" },
    { id: "neurons", label: "Neuron edit", color: "#267154", shape: "diamond" },
  ];
  const pairs = {
    de: ["German", "Dutch"],
    es: ["Spanish", "Portuguese"],
    fr: ["French", "Italian"],
  };
  const state = { target: "de", tau: 1 };
  const byId = id => document.getElementById(id);

  function svgNode(tag, attributes, content) {
    const node = document.createElementNS(NS, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    if (content !== undefined) node.textContent = content;
    return node;
  }

  function number(value) {
    if (value === 0) return "0";
    const a = Math.abs(value);
    const digits = a >= 0.1 ? 2 : a >= 0.01 ? 3 : Math.max(0, 1 - Math.floor(Math.log10(a)));
    return value.toFixed(digits).replace("-", "−");
  }

  function range(values) {
    const lo = number(Math.min(...values));
    const hi = number(Math.max(...values));
    return lo === hi ? lo : `${lo}–${hi}`;
  }

  function mark(svg, method, x, y, title) {
    const attrs = { fill: method.color, stroke: "var(--research-paper, white)", "stroke-width": 1 };
    let point;
    if (method.shape === "square") {
      point = svgNode("rect", { ...attrs, x: x - 4.5, y: y - 4.5, width: 9, height: 9 });
    } else if (method.shape === "triangle") {
      point = svgNode("path", { ...attrs, d: `M ${x} ${y - 6} l 5.5 10 h -11 z` });
    } else if (method.shape === "diamond") {
      point = svgNode("path", { ...attrs, d: `M ${x} ${y - 6} l 6 6 -6 6 -6 -6 z` });
    } else {
      point = svgNode("circle", { ...attrs, cx: x, cy: y, r: 5 });
    }
    point.append(svgNode("title", {}, title));
    svg.append(point);
  }

  function plot(container, groups, { title, subtitle, field, ticks, minimum, maximum, axis, reference }) {
    const isEnglish = field === "english_kl";
    const width = Math.max(240, container.clientWidth || 470);
    const compact = width < 450;
    const height = compact ? 380 : 306;
    const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
    const ink = "var(--research-ink, #182d3b)";
    const muted = "var(--research-muted, #526771)";
    const text = (x, y, content, attrs = {}) => svg.append(svgNode("text", {
      x, y, fill: ink, "font-size": 14, ...attrs,
    }, content));

    text(16, 24, title, { "font-size": 20, "font-weight": 600 });
    text(16, 46, subtitle, { fill: muted, "font-size": 13 });

    const left = compact ? (isEnglish ? 29 : 16) : 188;
    const right = width - (isEnglish ? 27 : 16);
    const scale = isEnglish ? Math.log10 : value => value;
    const x = value => left + (scale(value) - scale(minimum)) / (scale(maximum) - scale(minimum)) * (right - left);
    const top = compact ? 92 : 69;
    const bottom = height - 54;
    ticks.forEach(tick => {
      svg.append(svgNode("line", {
        x1: x(tick), x2: x(tick), y1: top, y2: bottom,
        stroke: "var(--research-rule, #dce2df)", "stroke-width": 1,
      }));
      text(x(tick), bottom + 20, String(tick), {
        "text-anchor": "middle", fill: muted, "font-size": isEnglish && compact ? 10 : 12,
      });
    });
    if (reference !== undefined) {
      svg.append(svgNode("line", {
        x1: x(reference), x2: x(reference), y1: top, y2: bottom,
        stroke: muted, "stroke-dasharray": "4 4", "stroke-width": 1.3,
      }));
    }

    groups.forEach(({ method, rows }, index) => {
      const values = rows.map(row => row[field]);
      const y = compact ? 105 + index * 66 : 89 + index * 46;
      if (compact) {
        svg.append(svgNode("rect", {
          x: 0, y: y - 41, width, height: 28, fill: "var(--research-paper, white)",
        }));
      }
      text(16, compact ? y - 22 : y - 4, method.label, { "font-size": compact ? 12 : 14 });
      text(compact ? width - 16 : 16, compact ? y - 22 : y + 15, range(values), {
        "text-anchor": compact ? "end" : "start", "font-weight": 600, "font-size": compact ? 12 : 14,
      });
      svg.append(svgNode("line", {
        x1: x(Math.min(...values)), x2: x(Math.max(...values)), y1: y, y2: y,
        stroke: method.color, "stroke-width": 2,
      }));
      rows.forEach(row => {
        const subset = method.id === "vpd_atom" ? "" : `, training subset ${row.draw + 1}`;
        mark(svg, method, x(row[field]), y, `${method.label}${subset}: ${number(row[field])} nat/token`);
      });
    });
    text(width / 2, height - 5, axis, { "text-anchor": "middle", fill: muted, "font-size": 12 });

    const values = groups.map(({ method, rows }) =>
      `${method.label}: ${range(rows.map(row => row[field]))} nat per token`).join(". ");
    const scaleDescription = isEnglish
      ? "Logarithmic axis from 0.00001 to 0.1."
      : "Both loss charts use an axis from zero to 1.2.";
    svg.setAttribute("aria-label", `${title}. ${subtitle}. ${values}. ${scaleDescription}`);
    container.replaceChildren(svg);
  }

  function drawTable(container, groups, captionTitle, fields) {
    const table = document.createElement("table");
    table.createCaption().textContent = captionTitle;
    const heading = table.createTHead().insertRow();
    ["Edit", ...fields.map(field => field.label)].forEach(label => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      heading.append(cell);
    });
    const body = table.createTBody();
    groups.forEach(({ method, rows }) => rows.forEach(edit => {
      const row = body.insertRow();
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = method.id === "vpd_atom" ? method.label : `${method.label}, subset ${edit.draw + 1}`;
      row.append(name);
      fields.forEach(({ key }) => {
        row.insertCell().textContent = number(edit[key]);
      });
    }));
    container.replaceChildren(table);
  }

  function decimal(value) {
    return value.toFixed(3).replace("-", "−");
  }

  function plotCalibration(calibration, target, tolerance) {
    const { coefficient, points, tau, validation_rise: validationRise } = calibration;
    const container = byId("selective-validation-plot");
    const width = Math.max(240, container.clientWidth);
    const height = 254;
    const left = 49;
    const right = width - 14;
    const top = 15;
    const bottom = height - 48;
    const ink = "var(--research-ink, #182d3b)";
    const muted = "var(--research-muted, #526771)";
    const rule = "var(--research-rule, #dce2df)";
    const accent = methods[0].color;
    const x = value => left + (value + 1) / 2 * (right - left);
    const y = value => bottom - value / 3.6 * (bottom - top);
    const svg = svgNode("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `${target} validation loss increase by tested subcomponent coefficient. ` +
        `The chosen coefficient ${decimal(coefficient)} raised loss by ${decimal(validationRise)} nat per token. ` +
        `The calibration target was ${tau} nat per token, with a tolerance of ${tolerance}.`,
    });
    const text = (x, y, content, attrs = {}) => svg.append(svgNode("text", {
      x, y, fill: muted, "font-size": 12, "text-anchor": "middle", ...attrs,
    }, content));

    for (const tick of [0, 1, 2, 3]) {
      svg.append(svgNode("line", { x1: left, x2: right, y1: y(tick), y2: y(tick), stroke: rule }));
      text(left - 8, y(tick) + 4, String(tick), { "text-anchor": "end" });
    }
    for (const tick of [-1, -0.5, 0, 0.5, 1]) {
      svg.append(svgNode("line", { x1: x(tick), x2: x(tick), y1: top, y2: bottom, stroke: rule }));
      text(x(tick), bottom + 20, String(tick).replace("-", "−"));
    }
    svg.append(
      svgNode("rect", {
        x: left, y: y(tau + tolerance), width: right - left,
        height: y(tau - tolerance) - y(tau + tolerance), fill: accent, "fill-opacity": 0.2,
      }),
      svgNode("line", {
        x1: left, x2: right, y1: y(tau), y2: y(tau), stroke: accent,
        "stroke-dasharray": "4 4", "stroke-opacity": 0.8,
      }),
    );
    for (const [tested, rise] of points) {
      const dot = svgNode("circle", { cx: x(tested), cy: y(rise), r: 3.2, fill: muted });
      dot.append(svgNode("title", {}, `Coefficient ${decimal(tested)}: ${decimal(rise)} nat/token`));
      svg.append(dot);
    }
    const chosen = svgNode("circle", {
      cx: x(coefficient), cy: y(validationRise), r: 6, fill: "var(--research-paper, white)",
      stroke: accent, "stroke-width": 2, "data-chosen": "",
    });
    chosen.append(svgNode("title", {}, `Chosen coefficient ${decimal(coefficient)}: ${decimal(validationRise)} nat/token`));
    svg.append(chosen);
    text((left + right) / 2, height - 5, "Subcomponent coefficient c", { fill: ink });
    const axisMiddle = (top + bottom) / 2;
    text(13, axisMiddle, "Validation loss increase (nat/token)", { fill: ink, transform: `rotate(-90 13 ${axisMiddle})` });
    container.replaceChildren(svg);
  }

  function showCalibration(calibration, atom, target, related, data) {
    const coefficient = decimal(calibration.coefficient);
    const validationRise = decimal(calibration.validation_rise);
    const targetRise = decimal(atom.target_rise);
    const relatedRise = number(atom.protected_rise);
    byId("selective-validation-heading").textContent = `${target} validation measurements`;
    byId("selective-validation-band").textContent = `Calibration target ±${data.tolerance}`;
    byId("selective-chosen-coefficient").textContent = `c = ${coefficient}`;
    byId("selective-coefficient-description").textContent = calibration.coefficient < 0
      ? "Reversed contribution" : "Reduced contribution";
    byId("selective-validation-language").textContent = `${target} loss rise`;
    byId("selective-validation-rise").textContent = `${validationRise} nat/token`;
    byId("selective-held-out-heading").textContent = `Held out · ${data.n_dates} dates`;
    byId("selective-target-language").textContent = `${target} loss rise`;
    byId("selective-target-rise").textContent = `${targetRise} nat/token`;
    byId("selective-related-language").textContent = `${related} loss rise`;
    byId("selective-related-rise").textContent = `${relatedRise} nat/token`;
    byId("selective-english-kl").textContent = `${number(atom.english_kl)} nat/token`;
    const table = document.createElement("table");
    table.createCaption().textContent = `Saved ${target} measurements at the ${calibration.tau} nat/token calibration level.`;
    const heading = table.createTHead().insertRow();
    for (const label of ["Coefficient", "Validation loss increase (nat/token)"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      heading.append(cell);
    }
    const body = table.createTBody();
    for (const [tested, rise] of calibration.points) {
      const row = body.insertRow();
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = decimal(tested) + (tested === calibration.coefficient ? " (chosen)" : "");
      row.append(name);
      row.insertCell().textContent = decimal(rise);
    }
    byId("selective-validation-table").replaceChildren(table);
  }

  function render(data, updateText) {
    const edits = data.edits.filter(edit => edit.target === state.target && edit.tau === state.tau);
    if (edits.length !== 12) throw new Error("Incomplete presentation data");
    const groups = methods.map(method => {
      const rows = edits.filter(edit => edit.method === method.id);
      return { method, rows: method.id === "vpd_atom" ? rows.slice(0, 1) : rows };
    });
    const [target, related] = pairs[state.target];
    const calibration = data.calibrations.find(row => row.target === state.target && row.tau === state.tau);
    if (!calibration) throw new Error("Missing validation calibration");
    plotCalibration(calibration, target, data.tolerance);
    const lossScale = {
      ticks: [0, 0.5, 1],
      minimum: 0,
      maximum: 1.2,
      axis: "Prediction loss increase (nat/token)",
    };
    plot(byId("visual-target-chart"), groups, {
      ...lossScale,
      title: target,
      subtitle: "Target language",
      field: "target_rise",
      reference: state.tau,
    });
    plot(byId("visual-protected-chart"), groups, {
      ...lossScale,
      title: related,
      subtitle: "Related language",
      field: "protected_rise",
    });
    plot(byId("visual-english-chart"), groups, {
      title: "English predictions",
      subtitle: `${target} edit · calibration ${state.tau} nat/token`,
      field: "english_kl",
      ticks: [0.00001, 0.0001, 0.001, 0.01, 0.1],
      minimum: 0.00001,
      maximum: 0.1,
      axis: "Mean KL divergence (nat/token, log scale)",
    });
    if (!updateText) return;

    const atom = groups[0].rows[0];
    showCalibration(calibration, atom, target, related, data);
    const comparators = groups.slice(1).flatMap(({ rows }) => rows);
    const targetRange = range(comparators.map(row => row.target_rise));
    const relatedRange = range(comparators.map(row => row.protected_rise));
    byId("visual-result-sentence").textContent =
      `On evaluation text, the VPD edit raised ${target} loss by ${number(atom.target_rise)} nat/token and ${related} loss by ${number(atom.protected_rise)} nat/token. ` +
      `The other edits raised ${target} loss by ${targetRange} nat/token and ${related} loss by ${relatedRange} nat/token.`;
    drawTable(byId("visual-results-table"), groups,
      `${target} and ${related} loss increases on evaluation text (nat/token). Calibration level: ${state.tau} nat/token.`, [
        { label: target, key: "target_rise" },
        { label: related, key: "protected_rise" },
      ]);
    drawTable(byId("visual-english-table"), groups,
      `Mean English KL on evaluation text for the ${target} edit at the ${state.tau} nat/token calibration level.`, [
        { label: "KL (nat/token)", key: "english_kl" },
      ]);
    document.querySelectorAll("[data-visual-tau]").forEach(button => {
      button.setAttribute("aria-pressed", String(Number(button.dataset.visualTau) === state.tau));
    });
    ["visual-pair", "visual-english-pair"].forEach(id => { byId(id).value = state.target; });
  }

  const toy = {
    remainder: [[2, -1, 3], [1, 3, -2], [-2, 1, 2]],
    u: [2, -1, 1],
    v: [2, -1, 1],
  };

  function matrixIllustration(slider) {
    const coefficient = Number(slider.value);
    const { remainder, u, v } = toy;
    const scaled = u.map(a => v.map(b => coefficient * a * b));
    const edited = remainder.map((row, i) => row.map((value, j) => value + scaled[i][j]));
    const integerOrTenth = value => String(Number(value.toFixed(1))).replace("-", "−");

    [remainder, scaled, edited].forEach((matrix, index) => {
      const group = byId(`visual-matrix-cells-${index}`);
      group.replaceChildren();
      matrix.forEach((row, i) => row.forEach((value, j) => {
        const x = 3 + j * 43;
        const y = 3 + i * 43;
        group.append(
          svgNode("rect", {
            x, y, width: 40, height: 40, rx: 2,
            fill: value < 0 ? "#826aab" : "#327e70",
            "fill-opacity": value === 0 ? 0.04 : 0.13 + Math.abs(value) * 0.055,
            stroke: "var(--research-rule, #dce2df)",
          }),
          svgNode("text", {
            x: x + 20, y: y + 21, fill: "var(--research-ink, #182d3b)",
            "font-size": 14, "font-weight": 600, "text-anchor": "middle", "dominant-baseline": "middle",
          }, integerOrTenth(value)),
        );
      }));
    });

    let summary = "Reduced contribution";
    if (coefficient === 1) summary = "Original matrix reconstructed";
    if (coefficient === 0) summary = "Subcomponent removed";
    if (coefficient < 0) summary = "Subcomponent sign reversed";
    const operation = [remainder, scaled, edited].map(matrix => integerOrTenth(matrix[0][0]));
    byId("visual-coefficient-value").value = integerOrTenth(coefficient);
    slider.setAttribute("aria-valuetext", `${coefficient}. ${summary}.`);
    byId("visual-matrix-state").textContent =
      `${summary}. Top-left entry: ${operation[0]}\u00a0+\u00a0(${operation[1]})\u00a0=\u00a0${operation[2]}.`;
    document.querySelector(".visual-matrix-equation").setAttribute("aria-label",
      `Toy matrix equation. The rest of the matrix plus the scaled subcomponent equals the edited matrix. ${summary}.`);
  }

  function init() {
    const slider = byId("visual-coefficient");
    matrixIllustration(slider);
    slider.addEventListener("input", () => matrixIllustration(slider));
    byId("visual-coefficient-control").hidden = false;
    byId("visual-matrix-key").hidden = false;
    if (window.ResearchOrbit && window.initMatrixOutput) window.initMatrixOutput(slider, toy);

    const interactive = [byId("visual-results-interactive"), byId("visual-english-interactive")];
    const fallback = [byId("visual-results-fallback"), byId("visual-english-fallback")];
    try {
      const data = {
        ...JSON.parse(byId("visual-selective-data").textContent),
        ...JSON.parse(byId("visual-selective-calibration").textContent),
      };
      interactive.forEach(container => { container.hidden = false; });
      render(data, true);
      fallback.forEach(container => { container.hidden = true; });

      ["visual-pair", "visual-english-pair"].forEach(id => byId(id).addEventListener("change", event => {
        state.target = event.target.value;
        render(data, true);
      }));
      document.querySelectorAll("[data-visual-tau]").forEach(button => button.addEventListener("click", () => {
        state.tau = Number(button.dataset.visualTau);
        render(data, true);
      }));

      let frame;
      const redraw = () => {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(() => render(data, false));
      };
      const widths = new Map();
      const containers = [
        "selective-validation-plot", "visual-target-chart", "visual-protected-chart", "visual-english-chart",
      ].map(byId);
      if (typeof ResizeObserver !== "undefined") {
        const observer = new ResizeObserver(entries => {
          for (const { target, contentRect } of entries) {
            if (widths.get(target) !== contentRect.width) {
              widths.set(target, contentRect.width);
              redraw();
            }
          }
        });
        containers.forEach(container => observer.observe(container));
      } else {
        window.addEventListener("resize", redraw);
      }
    } catch (error) {
      interactive.forEach(container => { container.hidden = true; });
      fallback.forEach(container => { container.hidden = false; });
      console.warn("Selective-editing visual could not initialize.", error);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
