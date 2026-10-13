"""Figures and tables from the selective-editing results.

Values come from the frozen analysis (confirm.py) and post-hoc analysis (posthoc.py) JSONs,
with rounding only for display. CE rises, KL and their differences are in nats per token. Every interval is a simultaneous 95% interval over resampled session dates, from
one of three families that never share a critical value: the frozen claim family, the frozen reference family and
the post-hoc dose-agreement family. Each method keeps one colour and marker in every figure. Confirm values are
filled markers, dev values open ones. SVG ids use a fixed hash salt, glyphs are drawn as paths and dates are
omitted for deterministic output.

    uv run python -m interesearch.selective_editing.figures results/selective_editing --out results/selective_editing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MAIN = "vpd_atom"
COMPARATORS = ("lora_rank1", "finetune_24", "neurons")  # claim comparators with an estimate (null_atom has none)
METHODS = (MAIN, *COMPARATORS)
REFERENCES = ("omp_finetune", "omp_lora", "ci_difference", "historical_atom")
LABEL = {
    "vpd_atom": "VPD subcomponent",
    "null_atom": "Randomized-dictionary control",
    "lora_rank1": "Rank-one LoRA",
    "finetune_24": "Full-matrix fine-tuning",
    "neurons": "Neuron edits",
    "omp_finetune": "OMP from full-matrix fine-tuning",
    "omp_lora": "OMP from rank-one LoRA",
    "ci_difference": "CI-difference selection",
    "historical_atom": "Previously chosen subcomponent",
    "whitened_rank1": "Whitened rank-one selection",
    "lora_rank1_long": "Rank-one LoRA, longer training",
    "finetune_24_long": "Full-matrix fine-tuning, longer training",
    "svd_atom": "SVD component",
}
COLOR = {"vpd_atom": "#2a78d6", "lora_rank1": "#c85224", "finetune_24": "#4a3aa7", "neurons": "#12865f"}
MARKER = {"vpd_atom": "o", "lora_rank1": "s", "finetune_24": "^", "neurons": "D"}
GREY, LIGHT = "#52514e", "#e4e3df"
TAUS = ("0.5", "1.0")
TARGET_FLAG = 0.1  # protocol.json target_deviation_flag: the frozen tolerance of one edit's target rise around tau
OFFSET = 0.17  # vertical distance of the confirm (above) and dev (below) marker from their row in the forest plot
WIDTH = 6.3  # inches: the A4 text width of the PDF
STYLE = {
    "svg.hashsalt": "interesearch-selective-editing", "svg.fonttype": "path", "font.family": "DejaVu Sans",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
    "axes.linewidth": 0.6, "xtick.color": GREY, "ytick.color": GREY, "axes.grid": True, "axes.grid.axis": "x",
    "grid.color": LIGHT, "grid.linewidth": 0.6, "lines.linewidth": 1.2,
}
MINUS = "−"


def signed(v: float, spec: str) -> str:
    return format(v, spec).replace("-", MINUS)


def interval(r: dict, spec: str = "+.3f", est: str = "estimate") -> str:
    return f"{signed(r[est], spec)} [{signed(r['lo'], spec)}, {signed(r['hi'], spec)}]"


def span(vs: list[float], spec: str) -> str:
    lo, hi = min(vs), max(vs)
    if format(lo, spec) == format(hi, spec):
        return signed(lo, spec)
    return f"{signed(lo, spec)}{' to ' if lo < 0 else '–'}{signed(hi, spec)}"


def one(vs: list) -> float:
    """The single value of a quantity that every draw shares."""
    if len(set(vs)) != 1:
        raise ValueError(f"expected one value, got {sorted(set(vs))}")
    return vs[0]


def rise_spec(vs: list[float]) -> str:
    """Three decimals, or four when every value is below 0.01 in size."""
    return ".4f" if max(abs(v) for v in vs) < 0.01 else ".3f"


def parse(contrast: str) -> tuple[str, str, str, str]:
    """'de->nl|tau=0.5|vpd_atom-lora_rank1' -> ('de', 'nl', '0.5', 'lora_rank1')."""
    pair, tau, methods = contrast.split("|")
    t, p = pair.split("->")
    return t, p, tau.removeprefix("tau="), methods.split("-", 1)[1]


def pair_text(t: str, p: str) -> str:
    return f"{t} → {p}"


def tau_text(tau: str) -> str:
    return f"τ = {float(tau):g}"


def _pairs(res: dict) -> dict[str, str]:
    return {t: p for t, p, _, _ in map(parse, (r["contrast"] for r in res["confirm_result"]["reading_table"]))}


def _edits(res: dict, split: str) -> list[dict]:
    return [e for e in res["posthoc"]["edits"] if e["split"] == split]


def _save(fig, path: Path) -> None:
    fig.savefig(path, format="svg", metadata={"Date": None}, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def _hdot(ax, x, y, lo, hi, color, marker, filled=True, ms=4.0):
    """One point with its interval as a plain horizontal bar."""
    ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt=marker, color=color, ms=ms, mfc=color if filled else "white",
                mew=1.0, elinewidth=1.0, capsize=0, zorder=3)


def effects_figure(res: dict, path: Path) -> None:
    """Achieved target CE rise against protected CE rise of every claim-method edit on the confirm dates, per
    target and draw, with the frozen tolerance tau +- 0.1 shaded."""
    edits = _edits(res, "confirm")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), sharex=True, sharey=True)
    for ax, (t, p) in zip(axes, _pairs(res).items()):
        for tau in map(float, TAUS):
            ax.axvspan(tau - TARGET_FLAG, tau + TARGET_FLAG, color=LIGHT, alpha=0.7, lw=0, zorder=0)
            ax.axvline(tau, color=GREY, lw=0.6, ls="--", zorder=1)
        for m in METHODS:
            es = [e for e in edits if e["target"] == t and e["method"] == m]
            ax.plot([e["target_rise"] for e in es], [e["protected_rise"] for e in es], ls="", marker=MARKER[m],
                    color=COLOR[m], ms=4.0, mew=0.8, alpha=0.85, zorder=3)
        ax.set_title(pair_text(t, p), loc="left")
        ax.set_xlabel("Target CE rise (nat/token)")
        ax.grid(axis="y")
    axes[0].set_xlim(0.3, 1.2)
    axes[0].set_xticks([0.5, 1.0])
    axes[0].set_ylim(bottom=-0.03)
    axes[0].set_ylabel("Protected CE rise (nat/token)")
    handles = [Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="", ms=4.0, label=LABEL[m]) for m in METHODS]
    handles.append(Patch(color=LIGHT, alpha=0.7, label="τ ± 0.1 (frozen tolerance)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.22),
               columnspacing=1.2, handletextpad=0.3)
    _save(fig, path)


def _rows(res: dict) -> list[tuple[str, str]]:
    return [(t, tau) for t in _pairs(res) for tau in TAUS]


def _claims(res: dict) -> dict[tuple[str, str, str], dict]:
    """(target, tau, comparator) -> its row of the confirm reading table."""
    out = {}
    for r in res["confirm_result"]["reading_table"]:
        t, _, tau, m = parse(r["contrast"])
        out[(t, tau, m)] = r
    return out


def contrasts_figure(res: dict, path: Path) -> None:
    """The 18 claim contrasts on confirm and dev: protected CE rise of vpd_atom minus the comparator's (frozen claim
    family, top) and the same difference in target CE rise (post-hoc dose-agreement family, bottom)."""
    claims = _claims(res)
    gaps = {s: res["posthoc"]["dose_agreement"][s]["rows"] for s in ("confirm", "dev")}
    rows, pairs = _rows(res), _pairs(res)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.1), sharey=True, gridspec_kw={"hspace": 0.55, "wspace": 0.12})
    gap_lim = 0.0
    for col, m in enumerate(COMPARATORS):
        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            ax.axvline(0, color=GREY, lw=0.6, zorder=1)
        bottom.axvspan(-0.1, 0.1, color=LIGHT, alpha=0.7, lw=0, zorder=0)
        lims = []
        for k, (t, tau) in enumerate(rows):
            r = claims[(t, tau, m)]
            name = f"{r['contrast']}|target_gap"
            for split, dy, filled in (("confirm", -OFFSET, True), ("dev", OFFSET, False)):
                c = r[split]
                _hdot(top, c["estimate"], k + dy, c["lo"], c["hi"], COLOR[m], MARKER[m], filled)
                g = gaps[split][name]
                _hdot(bottom, g["mean"], k + dy, g["lo"], g["hi"], COLOR[m], MARKER[m], filled)
                lims += [c["lo"], c["hi"]]
                gap_lim = max(gap_lim, abs(g["lo"]), abs(g["hi"]))
        lo, hi = min(lims), max(0.0, max(lims))
        top.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        comparator_label = LABEL[m][0].lower() + LABEL[m][1:]
        top.set_title(f"{LABEL[MAIN]} −\n{comparator_label}", loc="left")
        bottom.set_title(f"{LABEL[MAIN]} −\n{comparator_label}", loc="left")
    for ax in axes[1]:
        ax.set_xlim(-1.12 * gap_lim, 1.12 * gap_lim)
    axes[0, 1].set_xlabel("Difference in protected CE rise (nat/token), frozen claim family")
    axes[1, 1].set_xlabel("Difference in target CE rise (nat/token), post hoc dose-agreement family")
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(rows)), [f"{pair_text(t, pairs[t])}, {tau_text(tau)}" for t, tau in rows])
    axes[0, 0].set_ylim(len(rows) - 0.5, -0.5)
    handles = [Line2D([], [], color=GREY, marker="o", ls="-", lw=1.0, ms=4.0, label="Confirmation (58 dates)"),
               Line2D([], [], color=GREY, marker="o", ls="-", lw=1.0, ms=4.0, mfc="white", label="Development (64 dates)"),
               Patch(color=LIGHT, alpha=0.7, label="±0.1 nat (post hoc agreement band)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.5)
    _save(fig, path)


def effects_table(res: dict) -> str:
    """Per claim method and tau, the range over targets and draws of what each edit did on confirm, its English KL
    on both splits and its norm."""
    conf, dev = _edits(res, "confirm"), _edits(res, "dev")
    lines = ["| Method | τ | Target rise | Protected rise | English rise | English KL, C (10⁻³) | "
             "English KL, D (10⁻³) | Edit norm |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for m in METHODS:
        for tau in map(float, TAUS):
            c = [e for e in conf if e["method"] == m and e["tau"] == tau]
            d = [e for e in dev if e["method"] == m and e["tau"] == tau]
            prot, en = [e["protected_rise"] for e in c], [e["guards"]["confirm_en"] for e in c]
            kl_c = [1e3 * v for e in c for v in e["english_kl"].values()]
            kl_d = [1e3 * v for e in d for v in e["english_kl"].values()]
            lines.append(f"| {LABEL[m] if tau == 0.5 else ''} | {tau:g} | {span([e['target_rise'] for e in c], '.2f')} | "
                         f"{span(prot, rise_spec(prot))} | {span(en, rise_spec(en))} | {span(kl_c, '.2f')} | "
                         f"{span(kl_d, '.2f')} | {span([e['edit_norm'] for e in d], '.2f')} |")
    n = len({(e["target"], e["draw"]) for e in conf})
    n_dates = res["posthoc"]["dose_agreement"]["confirm"]["n_dates"]
    lines += ["", (f": What the edits did on the {n_dates} confirmation dates, as the range over {n} edits per row "
              "(3 targets × 3 draws, with the VPD subcomponent draws sharing one edit per target). Rises are CE rises in nat/token "
              "on the target, the protected language and English. C: confirmation. D: development. English KL is from the base model. "
              "On development data it covers the English validation set and the three guards. Edit norm: Frobenius norm of the "
              'weight change. {#tbl-effects tbl-colwidths="[13,4,10,14,17,14,14,10]"}'), ""]
    return "\n".join(lines)


def claim_table(res: dict) -> str:
    """The 18 claim contrasts: the frozen estimates on confirm and dev and the post-hoc target gap on confirm."""
    gaps = res["posthoc"]["dose_agreement"]["confirm"]["rows"]
    pairs, claims = _pairs(res), _claims(res)
    lines = ["| Pair | τ | Comparator | Protected, confirmation | Protected, development | Target gap, confirmation |",
             "|:--|:--|:--|:--|:--|:--|"]
    for t, tau in _rows(res):
        for i, m in enumerate(COMPARATORS):
            r = claims[(t, tau, m)]
            g = gaps[f"{r['contrast']}|target_gap"]
            pair = pair_text(t, pairs[t]) if i == 0 and tau == TAUS[0] else ""
            lines.append(f"| {pair} | {f'{float(tau):g}' if i == 0 else ''} | {LABEL[m]} | {interval(r['confirm'])} | "
                         f"{interval(r['dev'])} | {interval(g, est='mean')} |")
    lines += ["", (": The 18 claim contrasts, VPD subcomponent minus comparator, in nat/token, with simultaneous 95% intervals. "
              "Protected: difference in protected CE rise, the frozen claim family. Target gap: difference in "
              "achieved target CE rise, from the post hoc 54-coordinate family. "
              '{#tbl-claims tbl-colwidths="[9,4,14,24,24,25]"}'), ""]
    return "\n".join(lines)


def interventions_table(res: dict) -> str:
    """The six vpd_atom interventions: atom, dose and mask coefficient, norm and achieved effects. Every draw
    selected the same atom and calibration the same dose, so each value is shared by the three draws."""
    edits, pairs = [e for e in res["posthoc"]["edits"] if e["method"] == MAIN], _pairs(res)
    if any(e["coefficient"] != 1 - e["scale"] for e in edits):
        raise ValueError("the caption assumes that every selected atom has sign -1")
    lines = ["| Pair | Subcomponent | τ | Dose a | Coefficient | Edit norm | Target rise, D | Target rise, C | "
             "Protected rise, C |", "|:--|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for t, tau in _rows(res):
        es = {s: [e for e in edits if e["split"] == s and e["target"] == t and e["tau"] == float(tau)]
              for s in ("dev", "confirm")}
        v = {k: one([e[k] for e in es["dev"] + es["confirm"]]) for k in ("atom", "scale", "coefficient", "edit_norm")}
        first = tau == TAUS[0]
        lines.append(f"| {pair_text(t, pairs[t]) if first else ''} | {v['atom'] if first else ''} | "
                     f"{float(tau):g} | {v['scale']:.3f} | {signed(v['coefficient'], '.3f')} | {v['edit_norm']:.2f} | "
                     f"{one([e['target_rise'] for e in es['dev']]):.2f} | "
                     f"{one([e['target_rise'] for e in es['confirm']]):.2f} | "
                     f"{one([e['protected_rise'] for e in es['confirm']]):.4f} |")
    lines += ["", (": The six VPD subcomponent interventions, each shared by the three draws: dose a, the subcomponent's mask "
              "coefficient 1\u00a0−\u00a0a after the edit, edit norm |α| · ‖A‖ and the achieved CE rises in nat/token. C: confirmation. D: development. "
              '{#tbl-interventions tbl-colwidths="[9,13,5,9,12,9,13,14,16]"}'), ""]
    return "\n".join(lines)


def references_table(res: dict) -> str:
    """The reference family on confirm and dev, with each reference's achieved target rise on the draws each
    contrast used."""
    pairs = _pairs(res)
    splits = {"confirm": res["confirm_analysis"], "dev": res["dev_analysis"]}
    lines = ["| Pair | τ | Reference | Target rise | Protected, confirmation | Protected, development | Draws used |",
             "|:--|:--|:--|:--|:--|:--|:--|"]
    for t, tau in _rows(res):
        composition = splits["confirm"]["composition"]["references"]
        refs = [m for m in REFERENCES if f"{t}->{pairs[t]}|tau={tau}|{MAIN}-{m}" in composition]
        for m in refs:
            name = f"{t}->{pairs[t]}|tau={tau}|{MAIN}-{m}"
            comp = composition[name]
            if splits["dev"]["composition"]["references"][name]["used_draws"] != comp["used_draws"]:
                raise ValueError(f"{name}: dev and confirm used different draws")
            if comp["status"] != {0: "empty", 3: "complete"}.get(len(comp["used_draws"]), "conditional"):
                raise ValueError(f"{name}: the caption reads the status from the number of draws used")
            if comp["status"] == "empty":
                cells = ["no edit", "no estimate", "no estimate"]
            else:
                rs = {s: splits[s]["references"][name] for s in splits}
                rise = [[e["target_rise"] for e in _edits(res, s) if e["method"] == m and e["target"] == t
                         and e["tau"] == float(tau) and e["draw"] in comp["used_draws"]] for s in splits]
                cells = [" / ".join(f"{sum(r) / len(r):.2f}" for r in rise)]
                if all(r["estimate"] == r["lo"] == r["hi"] == 0.0 for r in rs.values()):
                    cells += ["0 (identical edit)", "0 (identical edit)"]
                else:
                    cells += [interval(r) for r in rs.values()]
            used = ", ".join(map(str, comp["used_draws"])) or "none"
            # Every row names its pair and τ: the PDF splits this table across pages.
            label = LABEL[m]
            lines.append(f"| {pair_text(t, pairs[t])} | {float(tau):g} | {label} | {' | '.join(cells)} | {used} |")
    lines += ["", (": Alternative subcomponent-selection comparisons. Target rise: the alternative edit's achieved target CE rise, averaged over "
              "the draws its contrast used, on confirmation / development data. The selected subcomponent's rise is in @tbl-interventions. "
              "Protected: VPD subcomponent minus reference, difference in protected CE rise, with simultaneous 95% intervals "
              "adjusted within this comparison family. These comparisons are descriptive. All values in nat/token. Draws used: the draws in which "
              "the reference has a matched, usable edit. With fewer than three the contrast is conditional, with "
              'none it has no estimate. {#tbl-references tbl-colwidths="[8,4,16,14,24,24,10]"}'), ""]
    return "\n".join(lines)


EXTENDED_COMPARISONS = ("whitened_rank1", "lora_rank1_long", "finetune_24_long", "svd_atom")  # the order of its table rows
EXCLUDED = {"matched (guard)": "failed the guard", "not reached on the evaluated grid": "not reached"}


def _no_estimate(excluded: list[str]) -> str:
    """'no estimate: 1 failed the guard, 2 not reached' from the excluded runs of a method row."""
    reasons = [EXCLUDED[e.split(": ", 1)[1]] for e in excluded]
    return "no estimate: " + ", ".join(f"{reasons.count(r)} {r}" for r in dict.fromkeys(reasons))


def extended_comparisons_table(res: dict) -> str:
    """The exploratory additional comparisons: every contrast in both families, with the runs behind it and the achieved
    target-test rises of both sides."""
    add, pairs = res["extended_comparisons_analysis"], _pairs(res)
    below = {"direction": "below 0", "within_band": False}
    if any(v != below for f in ("labels", "labels_stable_runs") for v in add[f].values()) or \
            add["stable_changes_a_label"]:
        raise ValueError("the caption assumes that every interval lies below 0 and outside the band")
    lines = ["| Pair | τ | Comparator | Runs (stable) | Protected, all runs | Protected, stable runs | Target rise, "
             "subcomponent / comparator |", "|:--|:--|:--|:--|:--|:--|:--|"]
    for t, tau in _rows(res):
        main = add["methods"][f"{t}|tau={tau}|{MAIN}"]
        for m in EXTENDED_COMPARISONS:
            name, row = f"{t}->{pairs[t]}|tau={tau}|{MAIN}-{m}", add["methods"][f"{t}|tau={tau}|{m}"]
            comp = add["composition"][name]
            if comp["main_runs"] != 3:
                raise ValueError(f"{name}: the caption assumes three runs of {MAIN}")
            if name in add["contrasts_absent"]:
                cells = [f"0 of {row['runs']}", _no_estimate(row["excluded"]), "", ""]
            else:
                learned = name in add["stable_runs"]
                runs = f"{comp['comparator_runs']}" + (f" ({comp['comparator_runs_stable']})" if learned else "")
                rise, stable = f"{row['target_test_mean']:.2f}", f"{row['target_test_mean_stable']:.2f}"
                cells = [runs, interval(add["all_runs"][name]), interval(add["stable_runs"][name]) if learned else "",
                         f"{main['target_test_mean']:.2f} / {rise}" + (f" ({stable})" if stable != rise else "")]
            # Every row names its pair and τ: the PDF splits this table across pages.
            lines.append(f"| {pair_text(t, pairs[t])} | {float(tau):g} | {LABEL[m]} | {' | '.join(cells)} |")
    n = add["all_runs"]["_n_groups"]
    lines += ["", (f": Exploratory comparisons on the {n} development dates: VPD subcomponent minus comparator, difference in "
              "protected CE rise in nat/token, with simultaneous 95% intervals. All runs and stable runs (learned "
              "updates without unstable runs) are separate families. Every interval lies below zero and none lies "
              f"within ±{add['band']:g}, in both families. Runs: the comparator's matched and usable runs (draws, "
              f"times three seeds for the learned updates), and in parentheses those that are stable. Each contrast "
              "includes all three VPD subcomponent draws. Target rise: the achieved target-test CE rise, mean over all runs, and in parentheses over stable runs where "
              "it differs. "
              '{#tbl-extended-comparisons tbl-colwidths="[8,4,18,7,23,23,17]"}'), ""]
    return "\n".join(lines)


FIGURES = {"effects.svg": effects_figure, "contrasts.svg": contrasts_figure}
TABLES = {"effects_table.md": effects_table, "claim_table.md": claim_table,
          "interventions_table.md": interventions_table, "references_table.md": references_table,
          "extended_comparisons_table.md": extended_comparisons_table}


def explainer_data(res: dict) -> dict:
    """Copy the saved confirm results for the interactive explainer, without rounding or analysis."""
    comparisons = []
    for row in res["confirm_result"]["reading_table"]:
        target, protected, tau, comparator = parse(row["contrast"])
        gap = res["posthoc"]["dose_agreement"]["confirm"]["rows"][f"{row['contrast']}|target_gap"]
        comparisons.append({
            "target": target, "protected_language": protected, "tau": float(tau), "comparator": comparator,
            "protected": {key: row["confirm"][key] for key in ("estimate", "lo", "hi")},
            "target_gap": {"estimate": gap["mean"], "lo": gap["lo"], "hi": gap["hi"]},
        })
    return {
        "description": "Presentation data copied from committed results. No new statistical analysis. "
                       "Numbers retain source precision.",
        "sources": {
            "edits": "results/selective_editing/posthoc.json#edits (confirm, four claim methods)",
            "protected_intervals": "results/selective_editing/confirm_result.json#reading_table",
            "target_gap_intervals": "results/selective_editing/posthoc.json#dose_agreement.confirm.rows",
        },
        "n_dates": res["confirm_result"]["n_groups"],
        "edits": [{key: edit[key] for key in ("target", "draw", "method", "tau", "target_rise", "protected_rise")}
                  for edit in _edits(res, "confirm") if edit["method"] in METHODS],
        "comparisons": comparisons,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Draw the selective-editing figures and tables from the results.")
    ap.add_argument("results", type=Path, help="results/selective_editing")
    ap.add_argument("--out", type=Path, required=True, help="results/selective_editing")
    a = ap.parse_args(argv)
    res = {f: json.loads((a.results / f"{f}.json").read_text())
           for f in ("confirm_result", "confirm_analysis", "dev_analysis", "posthoc", "extended_comparisons_analysis")}
    matplotlib.use("svg")
    (a.out / "figures").mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for name, draw in FIGURES.items():
            draw(res, a.out / "figures" / name)
    for name, make in TABLES.items():
        (a.out / name).write_text(make(res))
    (a.out / "explainer_data.json").write_text(json.dumps(explainer_data(res), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
