"""Figures and tables from the repair-transfer analysis.

Values come from the analysis JSON (analysis.py), with rounding only for display. Recovery is shown in percent,
differences of recovery in percentage points (pp), and every interval is the analysis's 95% date-bootstrap interval.
SVG ids use a fixed hash salt, glyphs are drawn as paths and dates are omitted for deterministic output.

    uv run python -m interesearch.repair_transfer.figures results/repair_transfer/analysis.json \
        --out results/repair_transfer
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .analysis import EQUIV_PAIRS, MAIN_CONTRASTS
from .costs import CURRENCIES
from .curve import CURVE_BUDGETS, CURVE_CAPS, curve_name
from .selection import PHI_MAX, RETAIN_TOL, RHO_HI, RHO_LO

TARGETS = {"primary_target": "Primary target", "replication_target": "Replication target"}
LABEL = {
    "strengths": "Scalar coefficients only",
    "strengths_best": "Scalar coefficients only + full-batch optimization",
    "donor_init": "Donor initialization",
    "donor_init_output": "Donor initialization, output only",
    "random_init": "Random initialization",
    "random_init_seed2": "Random initialization, second seed",
    "fresh": "Fresh",
    "fresh_seed2": "Fresh, second seed",
    "donor_fixed": "Fixed donor repair",
}
CHART_LABEL = {**LABEL, "strengths_best": "Scalar coefficients only\n+ full-batch optimization",
               "donor_init_output": "Donor initialization,\noutput only"}
COLOR = {"donor_init": "#2a78d6", "donor_init_output": "#12865f", "random_init": "#4a3aa7", "fresh": "#c85224",
         "strengths": "#52514e", "strengths_best": "#52514e", "donor_fixed": "#8a8984"}
MARKER = {"donor_init": "o", "donor_init_output": "D", "random_init": "s", "fresh": "^", "strengths": "v",
          "strengths_best": "v"}
BUDGET_TEXT = {"256": "256", "512": "512", "2K": "2,048"}
CURRENCY_TEXT = {"unique_tokens": "Unique tokens", "fit_token_presentations": "Fit-token presentations",
                 "gpu_s": "GPU seconds (estimate)"}
TRAINED = ("strengths", "donor_init", "donor_init_output", "random_init", "fresh")
# matched-cost contrasts of the warm-start figure: (minuend, subtrahend) -> colour, marker, filled
WARM_START = {("donor_init", "fresh"): (COLOR["donor_init"], "o", True),
              ("donor_init_output", "fresh"): (COLOR["donor_init_output"], "D", True),
              ("donor_init", "random_init"): (COLOR["random_init"], "s", False)}
REACH_FAMILIES = ("donor_init", "donor_init_output", "random_init", "fresh")
PRESENTATION_TICKS = (5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6)
WIDTH = 6.3  # inches: the A4 text width of the PDF
STYLE = {
    "svg.hashsalt": "interesearch-repair-transfer", "svg.fonttype": "path", "font.family": "DejaVu Sans",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.2,
}
MINUS = "−"


def pct(v: float, spec: str = ".1f") -> str:
    return format(100 * v, spec).replace("-", MINUS)


def interval(lo_hi, spec: str) -> str:
    return f"[{pct(lo_hi[0], spec)}, {pct(lo_hi[1], spec)}]"


def contrast_text(c: dict) -> str:
    """Tenths of a pp, retaining digits when rounding would hide a sign or band boundary."""
    values = [100 * v for v in (c["diff"], *c["ci"])]
    digits = 1
    while any(v != boundary and round(v, digits) == boundary
              for v in values for boundary in (0, -2, 2)):
        digits += 1
    spec = f"+.{digits}f"
    return f"{pct(c['diff'], spec)} {interval(c['ci'], spec)}"


def cell_short(name: str) -> str:
    """'donor_init@256:le50' -> '256 / ≤50'."""
    b, _, cap = name.split("@")[1].partition(":le")
    return f"{BUDGET_TEXT[b]} / {'≤' + cap if cap else 600}"


def _save(fig, path: Path) -> None:
    fig.savefig(path, format="svg", metadata={"Date": None}, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def _dot(ax, x, y, lo, hi, color, marker, filled=True, ms=4.2):
    """One point with its 95% interval as a plain bar (values in display units)."""
    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=marker, color=color, ms=ms, mfc=color if filled else "white",
                mew=1.0, elinewidth=1.0, capsize=0, zorder=3)


def recovery_figure(res: dict, path: Path) -> None:
    """Recovery of every trained family at each fit budget (600 steps), with the fixed donor repair as a band."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True)
    offsets = {f: (i - 2.5) * 0.13 for i, f in enumerate((*TRAINED, "strengths_best"))}
    for ax, (target, title) in zip(axes, TARGETS.items()):
        cells = res["targets"][target]["cells"]
        fixed = cells["donor_fixed"]
        lo, hi = fixed["ci_r_nl"]
        ax.axhspan(100 * lo, 100 * hi, color=COLOR["donor_fixed"], alpha=0.18, lw=0, zorder=1)
        ax.axhline(100 * fixed["point"]["r_nl"], color=COLOR["donor_fixed"], ls="--", lw=0.9, zorder=2)
        points = [(i, f, f"{f}@{b}") for i, b in enumerate(CURVE_BUDGETS) for f in TRAINED]
        for i, f, g in points + [(2, "strengths_best", "strengths_best@2K")]:
            c = cells[g]
            _dot(ax, i + offsets[f], 100 * c["point"]["r_nl"], *(100 * v for v in c["ci_r_nl"]), COLOR[f], MARKER[f],
                 filled=f != "strengths_best")
        ax.set_xticks(range(len(CURVE_BUDGETS)), [BUDGET_TEXT[b] for b in CURVE_BUDGETS])
        ax.set_xlim(-0.55, 2.55)
        ax.set_xlabel("Unique fit tokens of the target")
        ax.set_title(title, loc="left")
    axes[0].set_ylim(50, 100)
    axes[0].set_ylabel("Recovery of Dutch damage (%)")
    handles = [Line2D([], [], color=COLOR[f], marker=MARKER[f], ls="", ms=4.2, label=CHART_LABEL[f]) for f in TRAINED]
    handles += [Line2D([], [], color=COLOR["strengths_best"], marker="v", mfc="white", ls="", ms=4.2,
                       label=CHART_LABEL["strengths_best"] + " (2,048 only)"),
                Line2D([], [], color=COLOR["donor_fixed"], ls="--", lw=0.9,
                       label=LABEL["donor_fixed"] + ", no fitting")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.30),
               columnspacing=1.2, handletextpad=0.3)
    _save(fig, path)


def warm_start_figure(res: dict, path: Path) -> None:
    """Paired differences at matched cost: the same budget and step cap for both families of a contrast."""
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 2.9), sharex=True, sharey=True)
    steps = [c or 600 for c in CURVE_CAPS]
    for row, target in enumerate(TARGETS):
        contrasts = res["targets"][target]["contrasts"]
        for col, b in enumerate(CURVE_BUDGETS):
            ax = axes[row, col]
            ax.axhline(0, color="#52514e", lw=0.6, zorder=1)
            for pair, (color, marker, filled) in WARM_START.items():
                pts = [contrasts[f"{curve_name(pair[0], b, c)} - {curve_name(pair[1], b, c)}"] for c in CURVE_CAPS]
                ys = [100 * p["diff"] for p in pts]
                ax.plot(steps, ys, color=color, lw=1.0, zorder=2)
                for x, y, p in zip(steps, ys, pts):
                    _dot(ax, x, y, *(100 * v for v in p["ci"]), color, marker, filled, ms=3.8)
            ax.set_xscale("log")
            ax.set_xticks(steps, [str(s) for s in steps])
            ax.minorticks_off()
            if row == 0:
                ax.set_title(f"{BUDGET_TEXT[b]} fit tokens", loc="left")
            if row == 1:
                ax.set_xlabel("Step cap per run (log scale)")
            if col == 0:
                ax.set_ylabel(f"{TARGETS[target]}\ndifference (pp)")
    handles = [Line2D([], [], color=color, marker=marker, mfc=None if filled else "white", ms=3.8,
                      label=f"{CHART_LABEL[x]} {MINUS}\n{CHART_LABEL[y].lower()}")
               for (x, y), (color, marker, filled) in WARM_START.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.14))
    _save(fig, path)


def reach_figure(res: dict, path: Path) -> None:
    """Quality-cost curve: each grid cell's paired difference from the reference at its true target-side
    fit-token presentations. Family by colour and marker as in the other figures, fit budget by line style. Filled
    markers reach the reference, and a ring marks the two cells that set K*."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    budget_style = {"256": ":", "512": "--", "2K": "-"}
    ticks = [f"{t / 1e6:g}M" if t >= 1e6 else f"{t / 1e3:g}K" for t in PRESENTATION_TICKS]
    for ax, target in zip(axes, TARGETS):
        curve = res["targets"][target]["curve"]
        grid, ref = curve["grid"], curve["grid"][curve["reference"]]["r_nl"]
        ax.axhline(0, color="#52514e", lw=0.6, zorder=1)
        ax.axhline(-100 * curve["tolerance"], color="#52514e", lw=0.8, ls=":", zorder=1)
        for f in REACH_FAMILIES:
            for b in CURVE_BUDGETS:
                rows = [grid[curve_name(f, b, c)] for c in CURVE_CAPS]
                xs = [r["cost"]["fit_token_presentations"] for r in rows]
                ys = [100 * (r["r_nl"] - ref) for r in rows]
                ax.plot(xs, ys, color=COLOR[f], lw=0.9, ls=budget_style[b], alpha=0.8, zorder=2)
                for x, y, r in zip(xs, ys, rows):
                    _dot(ax, x, y, *(100 * v for v in r["vs_reference_ci"]), COLOR[f], MARKER[f],
                         filled=r["reaches"], ms=3.8)
        for f in ("fresh", "donor_init"):
            r = grid[curve["reach"][f]["fit_token_presentations"]["cell"]]
            x, y = r["cost"]["fit_token_presentations"], 100 * (r["r_nl"] - ref)
            # Keep the selected family's shape visible when another cell overlaps it.
            ax.plot(x, y, marker=MARKER[f], ms=3.8, color=COLOR[f], mew=1.0, ls="", zorder=4)
            ax.plot(x, y, "o", ms=9, mfc="none", mec="#1a1a1a", mew=0.9, zorder=5)
        ax.set_xscale("log")
        ax.set_xticks(PRESENTATION_TICKS, ticks)
        ax.minorticks_off()
        ax.set_xlabel("Fit-token presentations per target (log scale)")
        ax.set_title(TARGETS[target], loc="left")
        ax.text(ax.get_xlim()[1], -100 * curve["tolerance"], "reach threshold ", ha="right", va="bottom",
                fontsize=7, color="#52514e")
    axes[0].set_ylabel("Difference from the reference (pp)")
    grey = "#52514e"
    handles = [Line2D([], [], color=COLOR[f], marker=MARKER[f], ms=3.8, lw=1.0, label=CHART_LABEL[f])
               for f in REACH_FAMILIES]
    handles += [Line2D([], [], color=grey, marker="o", ls="", ms=3.8, label="reaches"),
                Line2D([], [], color=grey, marker="o", ls="", ms=3.8, mfc="white", label="does not reach")]
    handles += [Line2D([], [], color=grey, ls=ls, lw=0.9, label=f"{BUDGET_TEXT[b]} tokens")
                for b, ls in budget_style.items()]
    handles.append(Line2D([], [], color="#1a1a1a", marker="o", ls="", ms=9, mfc="none", mew=0.9,
                          label="cheapest reaching cell"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.27),
               columnspacing=1.2)
    _save(fig, path)


def _family(group: str) -> str:
    return group.split("@")[0]


def contrast_label(key: str) -> str:
    x, y = key.split(" - ")
    return f"{LABEL[_family(x)]} {MINUS} {LABEL[_family(y)].lower()}"


def secondary_table(res: dict) -> str:
    """Secondary contrasts of the 2,048-token cells (up to 600 steps) for both targets."""
    margin = pct(res["equivalence_margin"], "g")
    lines = [f"| Contrast (2,048 fit tokens) | Primary target | Replication target | Inside ±{margin} pp |",
             "|:--|:--|:--|:--|"]
    for x, y in MAIN_CONTRASTS:
        key = f"{x} - {y}"
        cells, band = [], []
        for target in TARGETS:
            t = res["targets"][target]
            c = t["contrasts"][key]
            same = t["cells"][x]["selected"] is not None and t["cells"][x]["selected"] == t["cells"][y]["selected"]
            cells.append("same repair selected" if same else contrast_text(c))
            band.append("yes" if c.get("equivalent") else "no")
        lines.append(f"| {contrast_label(key)} | {cells[0]} | {cells[1]} | "
                     f"{' / '.join(band) if (x, y) in EQUIV_PAIRS else '—'} |")
    lines += ["", (": Secondary contrasts at the full budget (2,048 unique fit tokens, up to 600 steps): paired difference "
              f"in recovery (pp) with its 95% interval, {res['targets']['primary_target']['n_dates']} readout dates "
              "per target. Rows marked '+ full-batch optimization' also include the two converged candidates. "
              "These intervals are descriptive, without multiplicity correction. The last column applies "
              "only to the five contrasts named in the protocol for an equivalence reading (primary / replication): "
              f"'yes' means that the whole interval lies inside the ±{margin} pp band. "
              '{#tbl-secondary tbl-colwidths="[40,24,24,12]"}'), ""]
    return "\n".join(lines)


def _amount(cur: str, v: float) -> str:
    return f"{v:,.1f}" if cur == "gpu_s" else f"{v:,.0f}"


def _k_star(b: dict) -> str:
    return b["reading"] if "k_star" not in b else f"{b['k_star']:.1f}"


def break_even_table(res: dict) -> str:
    """Per currency and target: the cheapest cells of fresh fitting and donor initialization that reach the
    reference, and the break-even number of targets K*."""
    lines = ["| Currency | Target | Fresh | Fresh cell | Donor init. | Donor init. cell | Break-even K\\* |",
             "|:--|:--|--:|:--|--:|:--|--:|"]
    for c in CURRENCIES:
        for target, title in TARGETS.items():
            curve = res["targets"][target]["curve"]
            cells = [f"{_amount(c, curve['reach'][f][c]['cost'])} | {cell_short(curve['reach'][f][c]['cell'])}"
                     for f in ("fresh", "donor_init")]
            name = CURRENCY_TEXT[c] if target == "primary_target" else ""
            lines.append(f"| {name} | {title.split()[0]} | {cells[0]} | {cells[1]} | "
                         f"{_k_star(curve['break_even']['donor_init'][c])} |")
    lines += ["", (": Target-side cost of the cheapest cell of fresh fitting and of donor initialization that reaches the reference "
              "(fresh at 2,048 tokens and up to 600 steps), found separately in each currency. A cell is written as unique "
              f"fit tokens / step cap. K\\* = donor cost / (fresh cost {MINUS} donor-initialization cost) is a modelled "
              "amortization: reuse pays off for more than K\\* targets. The donor's one-time cost and what each currency "
              "counts and leaves out are in the [cost-accounting appendix](#sec-app-cost). GPU seconds are an "
              "estimate of allocated shared work, and the donor's come from a different run. "
              '{#tbl-break-even tbl-colwidths="[17,13,10,14,10,13,23]"}'), ""]
    return "\n".join(lines)


def readout_table(res: dict) -> str:
    """Readout of every 2,048-token cell and of the fixed donor repair: recovery and the readout constraints."""
    groups = ("strengths", "strengths_best", "donor_init", "donor_init_output", "random_init", "random_init_seed2",
              "fresh", "fresh_seed2")
    lines = [("| Target | Repair | Recovery (%) [95% CI] | KL recovery | German effect kept | German drift | "
              "Worst retain change |"), "|:--|:--|:--|--:|--:|--:|--:|"]
    for target, title in TARGETS.items():
        cells = res["targets"][target]["cells"]
        for g in [f"{f}@2K" for f in groups] + ["donor_fixed"]:
            c, p = cells[g], cells[g]["point"]
            # Every row names its target: the PDF splits this table across pages.
            lines.append(f"| {title.split()[0]} | {LABEL[_family(g)]} | "
                         f"{pct(p['r_nl'])} {interval(c['ci_r_nl'], '.1f')} | "
                         f"{pct(p['r_kl'])} | {p['rho_de']:.3f} | {p['phi_de']:.4f} | "
                         f"{format(max(p['retain'].values()), '+.4f').replace('-', MINUS)} |")
    lines += ["", (": Readout of the 2,048-token cells (up to 600 steps) and the fixed donor repair on the "
              f"{res['targets']['primary_target']['n_dates']} readout dates. Rows marked '+ full-batch "
              "optimization' also include the two converged candidates. KL recovery (%) is the share of the "
              "fine-tune's Dutch KL from the base model that the repair removes. German effect kept is the ratio of "
              f"German CE damage with and without the repair (required {RHO_LO}–{RHO_HI}). German drift is the "
              "repair's German KL from the fine-tuned model relative to the base model's KL from the fine-tuned "
              f"model (required at most {PHI_MAX}). Worst retain change is the largest CE change in English, French, "
              f"Spanish or Italian against the fine-tuned model in nat/token (required at most +{RETAIN_TOL}). "
              '{#tbl-readout tbl-colwidths="[13,25,20,10,10,10,12]"}'), ""]
    return "\n".join(lines)


FIGURES = {"recovery.svg": recovery_figure, "warm_start.svg": warm_start_figure, "reach.svg": reach_figure}
TABLES = {"secondary_table.md": secondary_table, "break_even_table.md": break_even_table,
          "readout_table.md": readout_table}


def explainer_data(res: dict, source_bytes: bytes) -> dict:
    """Copy saved recovery cells and paired intervals for the explainer, retaining source precision."""
    targets = []
    for key, label in TARGETS.items():
        target = res["targets"][key]

        def cell(name: str) -> dict:
            saved = target["cells"][name]
            return {"source_cell": name, "recovery": saved["point"]["r_nl"], "ci": saved["ci_r_nl"]}

        full = [{"family": family, "label": text, **cell(name)} for family, text, name in (
            ("strengths", LABEL["strengths_best"], "strengths_best@2K"),
            ("donor_init", LABEL["donor_init"], "donor_init@2K"),
            ("fresh", LABEL["fresh"], "fresh@2K"),
        )]
        grid = []
        for budget in CURVE_BUDGETS:
            for cap in CURVE_CAPS:
                names = {family: curve_name(family, budget, cap) for family in ("strengths", "donor_init", "fresh")}
                contrast_name = f"{names['donor_init']} - {names['fresh']}"
                contrast = target["contrasts"][contrast_name]
                grid.append({
                    "budget": budget, "step_cap": cap if cap is not None else 600,
                    "strengths": cell(names["strengths"]), "donor": cell(names["donor_init"]),
                    "fresh": cell(names["fresh"]),
                    "gap": {"source_contrast": contrast_name, "diff": contrast["diff"], "ci": contrast["ci"],
                            "equivalent": contrast.get("equivalent")},
                })
        primary = target["primary"]
        targets.append({
            "key": key, "label": label, "n_dates": target["n_dates"],
            "primary": {"source_contrast": primary["contrast"], "diff": primary["diff"], "ci": primary["ci"]},
            "full": full, "grid": grid,
        })
    return {
        "schema_version": 1,
        "source": {
            "path": "results/repair_transfer/analysis.json", "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "study": res["study"],
            "uncertainty": "95% percentile bootstrap over evaluation dates, with paired intervals for differences",
            "n_boot": res["n_boot"],
            "units": "recovery and difference as fractions, rendered as percent and percentage points",
        },
        "equivalence_margin": res["equivalence_margin"],
        "budgets": [{"key": budget, "tokens": {"256": 256, "512": 512, "2K": 2048}[budget]}
                    for budget in CURVE_BUDGETS],
        "step_caps": [cap if cap is not None else 600 for cap in CURVE_CAPS],
        "targets": targets,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Draw the repair-transfer figures and tables from the analysis JSON.")
    ap.add_argument("analysis", type=Path, help="results/repair_transfer/analysis.json")
    ap.add_argument("--out", type=Path, required=True, help="results/repair_transfer")
    a = ap.parse_args(argv)
    source_bytes = a.analysis.read_bytes()
    res = json.loads(source_bytes)
    matplotlib.use("svg")
    (a.out / "figures").mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for name, draw in FIGURES.items():
            draw(res, a.out / "figures" / name)
    for name, make in TABLES.items():
        (a.out / name).write_text(make(res))
    (a.out / "explainer_data.json").write_text(json.dumps(explainer_data(res, source_bytes), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
