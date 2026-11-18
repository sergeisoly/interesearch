/* All estimates and intervals come from the saved analysis. */
(() => {
  "use strict";
  const METHODS = {
    strengths: { label: "Expanded scalar search", color: "#52514e" },
    donor_init: { label: "Donor initialization", color: "#2364a5" },
    fresh: { label: "First fresh start", color: "#a54520" }
  };
  const LOSS_AXIS_MAX = 0.20;
  const pct = value => (value * 100).toFixed(1);
  const signed = value => `${value < 0 ? "−" : "+"}${pct(Math.abs(value))}`;
  const interval = (bounds, format = pct) => `[${format(bounds[0])}, ${format(bounds[1])}]`;
  const comma = value => value.toLocaleString("en-US");
  const axisPosition = loss => `${100 * loss / LOSS_AXIS_MAX}%`;

  function append(parent, tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    parent.append(node);
    return node;
  }

  function column(row, label) {
    const cell = append(row, "th", "", label);
    cell.scope = "col";
  }

  function rowLabel(row, label) {
    const cell = append(row, "th", "", label);
    cell.scope = "row";
  }

  function initLossChart(data) {
    const chart = document.getElementById("repair-full-chart");
    const panels = document.getElementById("repair-loss-panels");
    const detail = document.getElementById("repair-loss-detail");
    const choices = [...chart.querySelectorAll("[data-repair-inspect]")];
    const tracks = [];
    for (const target of data.targets) {
      const panel = append(panels, "section", "repair-loss-panel");
      append(panel, "h3", "", target.label);
      const unrepaired = append(panel, "p", "repair-loss-before", "Unrepaired fine-tune");
      append(unrepaired, "strong", "", target.unrepaired_dutch_excess_loss.toFixed(4));
      for (const cell of target.full) {
        const method = METHODS[cell.family];
        const row = append(panel, "div", "repair-loss-row");
        const labels = append(row, "div", "repair-loss-label");
        append(labels, "span", "", method.label);
        append(labels, "span", "repair-loss-number", `${cell.dutch_excess_loss.toFixed(4)} (${pct(cell.recovery)}%)`);

        const track = append(row, "div", "repair-loss-track");
        track.setAttribute("aria-hidden", "true");
        track.style.setProperty("--repair-method-color", method.color);
        const baseline = append(track, "span", "repair-loss-baseline");
        baseline.style.left = axisPosition(target.unrepaired_dutch_excess_loss);
        const dot = append(track, "span", "repair-loss-dot");
        dot.style.left = axisPosition(cell.dutch_excess_loss);
        tracks.push({ row, family: cell.family });
      }

      const axis = append(panel, "div", "repair-loss-axis");
      axis.setAttribute("aria-hidden", "true");
      for (const value of [0, 0.05, 0.10, 0.15, LOSS_AXIS_MAX]) {
        const tick = append(axis, "span", "", value === 0 ? "0" : value.toFixed(2));
        tick.style.left = axisPosition(value);
      }
    }

    function inspect(family) {
      for (const button of choices) {
        button.setAttribute("aria-pressed", String(button.dataset.repairInspect === family));
      }
      for (const track of tracks) track.row.toggleAttribute("data-selected", track.family === family);
      const kept = data.targets.map(target => {
        const cell = target.full.find(candidate => candidate.family === family);
        return `${target.label} ${pct(cell.german_effect_kept)}%`;
      });
      detail.textContent = `German effect kept · ${kept.join(" · ")}`;
    }

    for (const button of choices) {
      button.addEventListener("click", () => inspect(button.dataset.repairInspect));
    }
    const [primary, replication] = data.targets.map(target => target.primary);
    document.getElementById("repair-full-primary").textContent =
      `Donor initialization improved recovery over the expanded scalar search by ${pct(primary.diff)} percentage ` +
      `points on the primary target (95% paired interval ${interval(primary.ci)}) and ${pct(replication.diff)} points ` +
      `on replication (${interval(replication.ci)}). Both intervals exceeded the study's ${primary.margin * 100}-point margin.`;
    inspect("donor_init");
    chart.hidden = false;
    document.getElementById("repair-full-data").open = false;
  }

  function initBudgetGrid(data) {
    const interactive = document.getElementById("repair-budget-interactive");
    const panels = document.getElementById("repair-budget-panels");
    const selection = document.getElementById("repair-budget-selected");
    const results = document.getElementById("repair-budget-detail");
    const buttons = [];
    const largestGap = Math.max(...data.targets.flatMap(target => target.grid.map(cell => Math.abs(cell.gap.diff))));
    const arrowMoves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };

    function select(budgetIndex, capIndex) {
      const budget = data.budgets[budgetIndex];
      const cap = data.step_caps[capIndex];
      const cells = data.targets.map(target =>
        target.grid.find(cell => cell.budget === budget.key && cell.step_cap === cap));
      for (const panelButtons of buttons) {
        for (const [row, rowButtons] of panelButtons.entries()) {
          for (const [col, button] of rowButtons.entries()) {
            const selected = row === budgetIndex && col === capIndex;
            button.setAttribute("aria-pressed", String(selected));
            button.tabIndex = selected ? 0 : -1;
          }
        }
      }

      const leads = cells.map((cell, index) => `${data.targets[index].label} ${signed(cell.gap.diff)} pp`);
      selection.textContent = `${comma(budget.tokens)} fit tokens · ${cap}-step cap`;
      append(selection, "span", "visually-hidden", `. ${leads.join(" · ")}`);

      const table = document.createElement("table");
      append(table, "caption", "", "Recovery with 95% date-bootstrap intervals. Donor advantages use paired intervals.");
      const head = table.createTHead().insertRow();
      column(head, "Measure");
      for (const target of data.targets) column(head, target.label);
      const body = table.createTBody();
      for (const [key, name] of [
        ["strengths", "Scalar-only recovery"],
        ["donor", "Donor recovery"],
        ["fresh", "First fresh recovery"],
        ["gap", "Donor advantage"]
      ]) {
        const row = body.insertRow();
        rowLabel(row, name);
        for (const cell of cells) {
          const value = key === "gap"
            ? `${signed(cell.gap.diff)} pp ${interval(cell.gap.ci, signed)}`
            : `${pct(cell[key].recovery)}% ${interval(cell[key].ci)}`;
          row.insertCell().textContent = value;
        }
      }
      results.replaceChildren(table);
    }

    for (const target of data.targets) {
      const table = append(panels, "table", "repair-budget-grid");
      append(table, "caption", "", target.label);
      const head = table.createTHead().insertRow();
      column(head, "Fit tokens");
      for (const cap of data.step_caps) column(head, String(cap));
      const body = table.createTBody();
      const panelButtons = [];
      buttons.push(panelButtons);
      for (const [budgetIndex, budget] of data.budgets.entries()) {
        const row = body.insertRow();
        rowLabel(row, comma(budget.tokens));
        const rowButtons = [];
        panelButtons.push(rowButtons);
        for (const [capIndex, cap] of data.step_caps.entries()) {
          const cell = target.grid.find(cell => cell.budget === budget.key && cell.step_cap === cap);
          const value = signed(cell.gap.diff);
          const button = append(row.insertCell(), "button", "", value);
          button.type = "button";
          button.setAttribute("aria-label",
            `${target.label}, ${comma(budget.tokens)} fit tokens, ${cap}-step cap, ${value} percentage points`);
          button.style.setProperty("--repair-cell-opacity", 0.06 + 0.23 * Math.abs(cell.gap.diff) / (largestGap || 1));
          rowButtons.push(button);
          button.addEventListener("click", () => select(budgetIndex, capIndex));
          button.addEventListener("keydown", event => {
            const move = arrowMoves[event.key];
            if (!move) return;
            event.preventDefault();
            const nextRow = budgetIndex + move[0];
            const nextCol = capIndex + move[1];
            const next = panelButtons[nextRow]?.[nextCol];
            if (next) {
              select(nextRow, nextCol);
              next.focus();
            }
          });
        }
      }
    }
    select(0, 0);
    interactive.hidden = false;
    document.getElementById("repair-budget-fallback").hidden = true;
  }

  function init() {
    try {
      const data = JSON.parse(document.getElementById("repair-explainer-data").textContent);
      initLossChart(data);
      initBudgetGrid(data);
    } catch (error) {
      console.warn("Repair-transfer figures could not initialize.", error);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
