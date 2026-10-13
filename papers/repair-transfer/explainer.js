/* Presentation only: all estimates and paired intervals come from the saved analysis. */
(() => {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const COLORS = { strengths: "#52514e", donor_init: "#2a78d6", fresh: "#c85224" };
  const INK = "#182d3b";
  const MUTED = "#526771";
  const GRID = "#dedfdc";
  const pct = value => (value * 100).toFixed(1);
  const signed = value => `${value >= 0 ? "+" : "−"}${Math.abs(value * 100).toFixed(1)}`;
  const interval = ci => `${pct(ci[0])}–${pct(ci[1])}`;
  const comma = value => value.toLocaleString("en-US");
  const familyLabel = family => ({ strengths: "Scalar coefficients only", donor_init: "Donor initialization", fresh: "Fresh initialization" })[family];

  function svgNode(tag, attrs = {}, content) {
    const el = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
    if (content !== undefined) el.textContent = content;
    return el;
  }

  function add(parent, tag, attrs, content) {
    const el = svgNode(tag, attrs, content);
    parent.append(el);
    return el;
  }

  function text(parent, x, y, content, attrs = {}) {
    return add(parent, "text", { x, y, fill: INK, "font-size": 14, ...attrs }, content);
  }

  function chart(container, width, height, title, description) {
    const svg = svgNode("svg", {
      viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img",
      "aria-labelledby": `${container.id}-title ${container.id}-description`,
      "font-family": "system-ui, -apple-system, sans-serif"
    });
    add(svg, "title", { id: `${container.id}-title` }, title);
    add(svg, "desc", { id: `${container.id}-description` }, description);
    container.replaceChildren(svg);
    return svg;
  }

  function whisker(svg, x1, x2, y, color) {
    add(svg, "line", { x1, x2, y1: y, y2: y, stroke: color, "stroke-width": 2 });
    [x1, x2].forEach(x => add(svg, "line", {
      x1: x, x2: x, y1: y - 5, y2: y + 5, stroke: color, "stroke-width": 1.5
    }));
  }

  function marker(svg, family, cx, cy) {
    if (family === "strengths") return add(svg, "path", {
      d: `M${cx},${cy + 5}l5,-9h-10Z`, fill: COLORS[family]
    });
    if (family === "fresh") return add(svg, "path", {
      d: `M${cx},${cy - 5}l5,9h-10Z`, fill: COLORS[family]
    });
    return add(svg, "circle", { cx, cy, r: 4.5, fill: COLORS[family] });
  }

  function init() {
    const embedded = document.getElementById("repair-explainer-data");
    if (!embedded) return;
    let data;
    try {
      data = JSON.parse(embedded.textContent);
    } catch (_) {
      document.getElementById("repair-curve-result").textContent =
        "The interactive figure could not load. The paper contains the same results as static figures.";
      return;
    }
    const full = document.getElementById("repair-full-chart");
    const curve = document.getElementById("repair-curve-chart");
    const targetSelect = document.getElementById("repair-target");
    const budgetSelect = document.getElementById("repair-budget");
    const stepInput = document.getElementById("repair-step");
    const methodCopy = {
      strengths: {
        title: "Same directions, new coefficients",
        text: "Keep the donor repair's input and output directions fixed. Learn one scalar coefficient per matrix, 24 numbers in all.",
        description: "The new arrow stays on the donor repair's line, but its scale can change.",
        path: "M58 177 L326 27 M315 28 L326 27 L321 37", label: [332, 25]
      },
      donor_init: {
        title: "Donor initialization",
        text: "Initialize both factors of every rank-one update from the donor repair, then fit both factors to the new target. The repair can change its directions as well as its magnitude.",
        description: "The new arrow can turn away from the donor's direction as the repair is fitted.",
        path: "M58 177 L348 124 M338 120 L348 124 L341 132", label: [275, 105]
      },
      fresh: {
        title: "Fresh initialization",
        text: "Begin with a zero update: a random input factor and a zero output factor. Train both factors on the new target.",
        description: "The donor arrow disappears. A new direction is learned from a zero-update start.",
        path: "M58 177 L333 89 M322 88 L333 89 L326 98", label: [278, 71]
      }
    };
    document.querySelectorAll("[data-repair-method]").forEach(button => button.addEventListener("click", () => {
      const method = button.dataset.repairMethod;
      const copy = methodCopy[method];
      document.querySelectorAll("[data-repair-method]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
      document.getElementById("repair-method-title").textContent = copy.title;
      document.getElementById("repair-method-description").textContent = copy.description;
      const arrow = document.getElementById("repair-learned-arrow");
      arrow.setAttribute("d", copy.path);
      arrow.setAttribute("stroke", COLORS[method]);
      const label = document.getElementById("repair-arrow-label");
      label.setAttribute("x", copy.label[0]);
      label.setAttribute("y", copy.label[1]);
      label.setAttribute("fill", INK);
      ["repair-donor-arrow", "repair-donor-endpoint", "repair-donor-label"].forEach(id =>
        document.getElementById(id).setAttribute("visibility", method === "fresh" ? "hidden" : "visible"));
      document.getElementById("repair-zero-start").setAttribute("visibility", method === "fresh" ? "visible" : "hidden");
      const heading = document.createElement("h3");
      const paragraph = document.createElement("p");
      heading.textContent = copy.title;
      paragraph.textContent = copy.text;
      document.getElementById("repair-method-copy").replaceChildren(heading, paragraph);
    }));

    function drawFull() {
      const width = Math.max(310, full.clientWidth || 760);
      const stacked = width < 650;
      const panelWidth = stacked ? width : (width - 36) / 2;
      const panelHeight = 322;
      const description = data.targets.map(t => `${t.label}: ${t.full.map(c =>
        `${familyLabel(c.family)}, ${pct(c.recovery)}% recovery, 95% interval ${interval(c.ci)}%`).join(". ")}`).join(". ");
      const svg = chart(full, width, stacked ? 2 * panelHeight : panelHeight,
        "Full-budget recovery on the two new fine-tunes", description);
      data.targets.forEach((target, index) => {
        const x0 = stacked ? 0 : index * (panelWidth + 36);
        const y0 = stacked ? index * panelHeight : 0;
        const left = x0 + 4;
        const right = x0 + panelWidth - 8;
        const scale = value => left + value * (right - left);
        text(svg, left, y0 + 22, target.label, { "font-size": 17, "font-weight": 650 });
        target.full.forEach((cell, row) => {
          const y = y0 + 65 + row * 72;
          text(svg, left, y, familyLabel(cell.family), { "font-size": 14 });
          if (cell.family === "strengths") text(svg, left, y + 18, "+ converged full-batch candidates", { "font-size": 12, fill: MUTED });
          const barY = y + (cell.family === "strengths" ? 25 : 12);
          text(svg, right, y, `${pct(cell.recovery)}%`, { "text-anchor": "end", "font-weight": 650 });
          add(svg, "rect", { x: left, y: barY, width: right - left, height: 14, fill: "#eaece9", rx: 2 });
          add(svg, "rect", {
            x: left, y: barY, width: scale(cell.recovery) - left, height: 14,
            fill: COLORS[cell.family], rx: 2
          });
          whisker(svg, scale(cell.ci[0]), scale(cell.ci[1]), barY + 7, INK);
        });
        [0, 0.5, 1].forEach(value => text(svg, scale(value), y0 + 284, `${value * 100}%`, {
          fill: MUTED, "font-size": 12, "text-anchor": value === 0 ? "start" : value === 1 ? "end" : "middle"
        }));
        text(svg, left, y0 + 310, "Share of Dutch damage recovered", { fill: MUTED, "font-size": 12 });
      });
      const primaryText = data.targets.map((target, i) =>
        `${signed(target.primary.diff)} pp (95% CI ${interval(target.primary.ci)}) on the ${i ? "replication" : "primary"} target`).join(" and ");
      document.getElementById("repair-primary-result").textContent =
        `Allowing directions to change adds ${primaryText}, compared with fitting scalar coefficients only.`;
    }

    function drawCurve() {
      const target = data.targets.find(t => t.key === targetSelect.value);
      const budget = data.budgets.find(b => b.key === budgetSelect.value);
      const step = data.step_caps[Number(stepInput.value)];
      const points = target.grid.filter(cell => cell.budget === budget.key);
      const current = points.find(cell => cell.step_cap === step);
      const width = Math.max(310, curve.clientWidth || 760);
      const stackedLegend = width < 650;
      const height = stackedLegend ? 410 : 375;
      const left = 44, right = width - 19, top = stackedLegend ? 92 : 53, bottom = height - 45;
      const x = cap => left + Math.log(cap / 50) / Math.log(600 / 50) * (right - left);
      const y = recovery => bottom - (recovery - 0.70) / 0.32 * (bottom - top);
      const svg = chart(curve, width, height,
        `${target.label}, ${comma(budget.tokens)} fit tokens: recovery by step cap`,
        points.map(p => `At ${p.step_cap} steps, scalar coefficients only ${pct(p.strengths.recovery)}%, donor initialization ${pct(p.donor.recovery)}%, fresh ${pct(p.fresh.recovery)}%.`).join(" ") +
        " Whiskers are 95% date-bootstrap intervals. The vertical scale begins at 70%.");
      ["strengths", "donor_init", "fresh"].forEach((family, index) => {
        const xp = stackedLegend ? left : left + index * (right - left) / 3;
        const yp = stackedLegend ? 17 + index * 23 : 17;
        add(svg, "line", { x1: xp, x2: xp + 13, y1: yp, y2: yp, stroke: COLORS[family], "stroke-width": 3 });
        marker(svg, family, xp + 6.5, yp);
        text(svg, xp + 19, yp + 4, familyLabel(family), { "font-size": 13 });
      });
      for (const value of [0.70, 0.80, 0.90, 1]) {
        add(svg, "line", { x1: left, x2: right, y1: y(value), y2: y(value), stroke: GRID, "stroke-width": 1 });
        text(svg, left - 9, y(value) + 4, `${Math.round(value * 100)}%`, {
          "text-anchor": "end", "font-size": 12, fill: MUTED
        });
      }
      add(svg, "line", {
        x1: x(step), x2: x(step), y1: top, y2: bottom, stroke: MUTED, "stroke-dasharray": "3 5", opacity: 0.65
      });
      data.step_caps.forEach(cap => text(svg, x(cap), bottom + 24, String(cap), {
        "text-anchor": "middle", "font-size": 12, fill: cap === step ? INK : MUTED,
        "font-weight": cap === step ? 700 : 400
      }));
      [["strengths", "strengths"], ["donor", "donor_init"], ["fresh", "fresh"]].forEach(([key, family]) => {
        add(svg, "path", {
          d: points.map((p, i) => `${i ? "L" : "M"}${x(p.step_cap)},${y(p[key].recovery)}`).join(" "),
          fill: "none", stroke: COLORS[family], "stroke-width": 2.5, "stroke-linejoin": "round"
        });
        points.forEach(p => {
          const cx = x(p.step_cap), cy = y(p[key].recovery);
          add(svg, "line", {
            x1: cx, x2: cx, y1: y(p[key].ci[0]), y2: y(p[key].ci[1]),
            stroke: COLORS[family], "stroke-width": 1.5
          });
          for (const limit of p[key].ci) add(svg, "line", {
            x1: cx - 4, x2: cx + 4, y1: y(limit), y2: y(limit), stroke: COLORS[family], "stroke-width": 1.5
          });
          if (p.step_cap === step) add(svg, "circle", {
            cx, cy, r: 8, fill: "#faf9f5", stroke: COLORS[family], "stroke-width": 1.5
          });
          const point = marker(svg, family, cx, cy);
          add(point, "title", {}, `${familyLabel(family)}, ${p.step_cap} steps: ${pct(p[key].recovery)}%, 95% CI ${interval(p[key].ci)}%.`);
        });
      });
      text(svg, (left + right) / 2, height - 2, "Step cap per run", {
        "text-anchor": "middle", "font-size": 12, fill: MUTED
      });
      document.getElementById("repair-step-value").textContent = `${step} steps`;
      stepInput.setAttribute("aria-valuetext", `${step} steps per run`);
      document.getElementById("repair-curve-result").textContent =
        `With ${comma(budget.tokens)} tokens and up to ${step} steps, recovery with donor initialization is ${pct(current.donor.recovery)}% versus ${pct(current.fresh.recovery)}% for fresh: ${signed(current.gap.diff)} pp (95% CI ${interval(current.gap.ci)}).`;
      const fullBudget = budget.key === "2K" && step === 600;
      document.getElementById("repair-curve-interpretation").textContent = fullBudget
        ? `The paired 95% interval lies within the ±${data.equivalence_margin * 100} percentage-point equivalence band.`
        : "";
      drawTable(points, target.label, budget.tokens);
    }

    function drawTable(points, targetLabel, tokens) {
      const table = document.createElement("table");
      const caption = document.createElement("caption");
      caption.textContent = `${targetLabel}, ${comma(tokens)} unique fit tokens. Values in brackets are 95% confidence intervals.`;
      table.append(caption);
      const head = table.createTHead().insertRow();
      ["Step cap", "Scalar coefficients only (%)", "Donor initialization (%)", "Fresh initialization (%)", "Paired advantage (pp)"].forEach(label => {
        const th = document.createElement("th");
        th.scope = "col";
        th.textContent = label;
        head.append(th);
      });
      const body = table.createTBody();
      points.forEach(p => {
        const row = body.insertRow();
        const label = document.createElement("th");
        label.scope = "row";
        label.textContent = String(p.step_cap);
        row.append(label);
        [`${pct(p.strengths.recovery)} [${interval(p.strengths.ci)}]`,
          `${pct(p.donor.recovery)} [${interval(p.donor.ci)}]`,
          `${pct(p.fresh.recovery)} [${interval(p.fresh.ci)}]`,
          `${signed(p.gap.diff)} [${interval(p.gap.ci)}]`].forEach(value => {
          row.insertCell().textContent = value;
        });
      });
      document.getElementById("repair-curve-table").replaceChildren(table);
    }

    [targetSelect, budgetSelect].forEach(control => control.addEventListener("change", drawCurve));
    stepInput.addEventListener("input", drawCurve);
    drawFull();
    drawCurve();
    let frame;
    const redraw = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => { drawFull(); drawCurve(); });
    };
    if (typeof ResizeObserver !== "undefined") {
      let lastWidth = full.clientWidth;
      new ResizeObserver(() => {
        if (full.clientWidth !== lastWidth) {
          lastWidth = full.clientWidth;
          redraw();
        }
      }).observe(full);
    } else {
      window.addEventListener("resize", redraw);
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
