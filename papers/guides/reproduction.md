---
image: ../social-card.png
description: "Setup, data, validated environments and paper builds."
title: "Code and data"
format: html
---

The code and saved measurements are in the [GitHub repository](https://github.com/sergeisoly/interesearch).

::: {.guide-task-options}
| Task | What it uses and produces | Instructions |
|---|---|---|
| Replay the analysis | Saved measurements become tables and figures; CPU, no model download | [Selective editing](selective-editing.md#replay-the-papers-analysis) · [Repair transfer](repair-transfer.md#replay-the-papers-analysis) |
| Run the model procedures | Model assets and token windows produce new measurements; requirements differ by study | [Selective editing](selective-editing.md) · [Repair transfer](repair-transfer.md) |
| Build the pages and papers | Committed results become HTML and PDF; Quarto and the PDF dependencies below | [Build the papers](#build-the-papers) |
:::

The shared Python setup follows. Each study guide describes its available model procedures and reproduction limits.

## Setup

From the root of an `interesearch` checkout (where `pyproject.toml` and `uv.lock` are), install
[uv](https://docs.astral.sh/uv/) and create the locked environment:

```bash
uv sync --locked
```

Run the study-guide commands from that directory.

Each study's `recipe.yaml` under `src/interesearch/` contains its published numerical settings. Model-stage
commands accept `--recipe PATH` for a complete replacement; explicit command-line settings take precedence.
Runs save the resolved settings before model computation. Analysis replay uses the settings saved with its
measurements; recognized older exports use the fixed published profile for missing settings.

Model downloads need `curl`. The Goodfire param-decomp environment also needs `git`.

## Data and reproduction limits

Saved measurements, calibration and training records, token windows and the donor repair are in `data/`.
Loaders check exported data against `SHA256SUMS`. Each `windows.json` identifies the text source, tokenizer
and selection rule.

Analysis replays use saved measurements and run without a model download. They matched `results/`
byte for byte in the reproduction checks. Tests allow relative error of 1e-12 and absolute error of 1e-15 for platform differences. Tables and
figures read these results. Numbers in the paper prose are maintained separately.

Model and decomposition downloads are checked against pinned sizes and hashes. To recheck them:

```bash
uv run python -m interesearch.assets verify assets
```

The repository includes measurements for the selective-editing confirmatory analysis. Rerunning its model
evaluation requires token windows and texts that are not published. The exploratory comparisons also include
saved measurements for analysis replay; a new model run retrains their learned updates, which were not retained.

## Validated environments

Analysis replay and paper builds were checked on an Apple M4 Mac. The locked Python environment is the reference
for those checks. The CPU quickstart was also checked on an AMD EPYC 9454 and needed about 3.3 GB of memory.

For software 0.2.0, the six analysis and figure commands reproduced all 23 published result files byte for byte
on the checked Mac. The test suite also passed with tiny models and without downloaded model assets.

The following model checks were performed for software release `v0.1.0`, before the module reorganization.
That code was compared with the implementation used for the published runs, on matched devices and runtimes.
The model and weight-edit operations, repair fitting and selective-editing steps I compared agreed bit for bit on CPU. On NVIDIA A10 with torch 2.8.0+cu128, the repair
fine-tune, reduced repair blocks and curvature record agreed bit for bit. Curvature records also agreed on L4.
On L40S, the hardware of the published runs, curvature and the first German selection agreed, including a null
dictionary check. These checks covered selected operations and reduced runs; complete experiments were not rerun in every environment.

Selective-editing label sampling depends on the GPU model. Its curvature gate requires the same GPU model,
SM count and torch/CUDA version as the selection run. The gate passed on L40S. It failed a float64
batch-invariance criterion on A10 and L4 in both implementations, so selection did not run there. The cause is
not localized. Changing hardware may therefore change selections or prevent a run.

## Build the papers

Install [Quarto](https://quarto.org/). HTML needs Quarto alone. PDF rendering also needs a TeX distribution with
LuaLaTeX, `rsvg-convert` and the STIX Two Text and STIX Two Math fonts.

```bash
cd papers
quarto render
```

The site is written to `papers/_site/` using committed results. For HTML only, use `quarto render --to html`.
The checked PDF toolchain was Quarto 1.10.19, LuaHBTeX 1.24.0 and `rsvg-convert` 2.63.2.

## Code map

Paths are relative to `src/interesearch/` in the repository. The study guides give the runnable commands.

| Concept | Module | Paper section |
|---|---|---|
| Subcomponent selection and curvature gate | `selective_editing/method/selector.py`, `fisher.py`, `runtime.py` | [Selection](../selective-editing/index.qmd#sec-selection) |
| Effect calibration | `selective_editing/method/calibration.py` | [Calibration](../selective-editing/index.qmd#sec-calibration) |
| Learned comparators | `selective_editing/method/finetune.py` | [Comparators](../selective-editing/index.qmd#sec-comparators) |
| Intervals and confirmatory verdicts | `selective_editing/analysis/statistics.py`, `confirm.py` | [Analysis](../selective-editing/index.qmd#sec-analysis), [contrasts](../selective-editing/index.qmd#sec-contrasts) |
| Additional comparisons | `selective_editing/analysis/extended.py` | [Longer training and alternative directions](../selective-editing/index.qmd#sec-extended-comparisons) |
| Target fine-tuning | `repair_transfer/method/finetune.py` | [Setup](../repair-transfer/index.qmd#sec-setup) |
| Repair parameterizations and fitting | `repair_transfer/method/fitting.py` | [Setup](../repair-transfer/index.qmd#sec-setup) |
| Feasibility and checkpoint selection | `repair_transfer/method/selection.py` | [Method details](../repair-transfer/index.qmd#sec-app-methods) |
| Recovery, intervals and primary reading | `repair_transfer/analysis/recovery.py`, `stats.py` | [Primary checks](../repair-transfer/index.qmd#sec-app-checks) |
| Cost accounting and reach | `repair_transfer/analysis/costs.py`, `curve.py` | [Cost accounting](../repair-transfer/index.qmd#sec-app-cost) |

Within each study, `method/` contains the numerical procedures, `analysis/` replays saved measurements,
and `stages/` coordinates model runs. `figures.py` generates tables, plots and explainer data. Tables are in
`results/<study>/tables/`, figures in `figures/`, and analysis JSON files remain directly under `results/<study>/`.

## Check the code

```bash
uv run --locked pytest -q
```

The suite needs no downloaded model assets. It checks analysis replay, weight edits and model paths on a tiny
model, input separation, and the file bindings used by the experiment runners.

## Sources and licences

The model, decomposition and labels come from Goodfire's
[Interpreting Language Model Parameters](https://www.goodfire.com/research/interpreting-lm-parameters).
The model implementation is adapted from [param-decomp](https://github.com/goodfire-ai/param-decomp).

Text windows come from Europarl v7, WikiText-2, NTREX-128 and Pile-uncopyrighted. `NOTICE` in the repository gives
their authors, revisions and terms. `LICENSING.md` maps licences to the code, research outputs and upstream material.
